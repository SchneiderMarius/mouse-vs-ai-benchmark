"""
track2_eval/src/activation_extraction.py
==========================================
ONNX graph manipulation and activation extraction.

All functions below are lifted VERBATIM from
Track2_avoid_kernel_death2.ipynb Cell 14.
DO NOT change logic — these functions must reproduce the official scores.

Functions
---------
keep_layer                          Filter out non-informative tensor names
expose_all_layer_outputs            Add all intermediate tensors as graph outputs
expose_layer_subset                 Add only a named subset as graph outputs
run_and_collect_subset              Run forward pass and collect activations
analyze_onnx_visual_encoder_boundary  Find the visual/policy fusion node
get_visual_encoder_layers_only      Return only pre-fusion layer names
"""

import io

import numpy as np
import onnx
import onnxruntime as ort
import tqdm
from onnx import TensorProto, shape_inference


# ---------------------------------------------------------------------------
# Layer name filter (verbatim from Cell 14)
# ---------------------------------------------------------------------------

def keep_layer(name: str) -> bool:
    ignore = [
        "version_number",
        "memory_size",
        "continuous_action",
        "_continuous_distribution",
        "Mul_output_0",
        "Add_output_0",
    ]
    return all(tok not in name for tok in ignore)


# ---------------------------------------------------------------------------
# Graph output exposure (verbatim from Cell 14)
# ---------------------------------------------------------------------------

def expose_all_layer_outputs(
    onnx_path: str,
    ranks: tuple = (2, 3, 4),
    dtypes: tuple = (TensorProto.FLOAT, TensorProto.FLOAT16),
) -> tuple:
    """
    Expose all valid intermediate outputs in the ONNX graph.

    Returns
    -------
    (session, layer_names)
        session    : onnxruntime.InferenceSession
        layer_names: list of str — all exposed output names
    """
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)

    existing = {o.name for o in model.graph.output}
    new_outs = []

    for v in model.graph.value_info:
        tt = v.type.tensor_type
        if (
            tt.elem_type in dtypes
            and len(tt.shape.dim) in ranks
            and v.name not in existing
        ):
            new_outs.append(v)

    model.graph.output.extend(new_outs)

    buf = io.BytesIO()
    onnx.save(model, buf)
    buf.seek(0)

    sess = ort.InferenceSession(
        buf.read(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return sess, [o.name for o in sess.get_outputs()]


def expose_layer_subset(
    onnx_path: str,
    subset_layers: list,
    ranks: tuple = (2, 3, 4),
    dtypes: tuple = (TensorProto.FLOAT, TensorProto.FLOAT16),
) -> ort.InferenceSession:
    """
    Return a session exposing only a specified subset of layers.
    """
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)

    existing = {o.name for o in model.graph.output}
    subset_set = set(subset_layers)
    new_outs = []

    for v in model.graph.value_info:
        if v.name in subset_set:
            tt = v.type.tensor_type
            if (
                tt.elem_type in dtypes
                and len(tt.shape.dim) in ranks
                and v.name not in existing
            ):
                new_outs.append(v)

    model.graph.output.extend(new_outs)

    buf = io.BytesIO()
    onnx.save(model, buf)
    buf.seek(0)

    sess = ort.InferenceSession(
        buf.read(),
        providers=["CPUExecutionProvider"],
    )
    return sess


# ---------------------------------------------------------------------------
# Forward pass (verbatim from Cell 14)
# ---------------------------------------------------------------------------

def run_and_collect_subset(
    session: ort.InferenceSession,
    layers: list,
    frames: np.ndarray,
    batch: int = 128,
) -> dict:
    """
    Run the model on all frames in batches and collect activations.

    Parameters
    ----------
    session : InferenceSession
    layers  : list of output names to collect
    frames  : (T, 1, H, W) float32
    batch   : batch size

    Returns
    -------
    dict {layer_name: (T, D) float32}  — D is the flattened spatial dimension
    """
    feats = {n: [] for n in layers}
    inp = session.get_inputs()[0].name

    for i in tqdm.trange(0, len(frames), batch, desc="forward", leave=False):
        outs = session.run(layers, {inp: frames[i : i + batch].astype("float32")})
        for n, a in zip(layers, outs):
            feats[n].append(a.reshape(a.shape[0], -1))

    return {n: np.concatenate(v, axis=0) for n, v in feats.items()}


# ---------------------------------------------------------------------------
# Visual encoder boundary detection (verbatim from Cell 14)
# ---------------------------------------------------------------------------

def analyze_onnx_visual_encoder_boundary(path: str) -> tuple:
    """
    Find the first ONNX graph node where visual features fuse with body/policy
    features.

    Logic (verbatim from Cell 14):
      Stop at the first node whose inputs contain BOTH:
        * an 'observation_encoder' tensor
        * a 'body_encoder' / 'body_endoder' / 'mlp_extractor' / 'policy' tensor

    Returns
    -------
    (pre_fusion_outputs, fusion_node_name)
        pre_fusion_outputs : list of str — all node output names before the
                             fusion node
        fusion_node_name   : str or None — name of the fusion node itself
    """
    model = onnx.load(path)
    graph = model.graph

    pre_fusion_outputs = []
    fusion_node_name = None

    for node in graph.node:
        inputs = list(node.input)

        found_obs = any("observation_encoder" in i for i in inputs)
        found_body = any(
            ("body_encoder" in i)
            or ("body_endoder" in i)   # note: typo in original, preserved verbatim
            or ("mlp_extractor" in i)
            or ("policy" in i)
            for i in inputs
        )

        if found_obs and found_body:
            fusion_node_name = node.name
            break

        for out in node.output:
            pre_fusion_outputs.append(out)

    return pre_fusion_outputs, fusion_node_name


def get_visual_encoder_layers_only(
    model_path: str,
    all_layers: list,
) -> tuple:
    """
    Restrict exposed ONNX outputs to layers before the policy/body fusion point.

    Fallback (verbatim from Cell 14): if < 3 pre-fusion matches are found,
    use the first half of valid layers.

    Returns
    -------
    (ve_layers, fusion_node_name)
    """
    pre_fusion_outputs, fusion_node_name = analyze_onnx_visual_encoder_boundary(
        model_path
    )

    ve_layers = [n for n in all_layers if n in set(pre_fusion_outputs)]

    if len(ve_layers) < 3:
        print(
            "  WARNING: Could not robustly match pre-fusion outputs. "
            "Falling back to first half of valid layers."
        )
        ve_layers = all_layers[: max(1, len(all_layers) // 2)]

    return ve_layers, fusion_node_name
