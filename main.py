"""
main.py — Hieroglyph Recognition & Translation System
Pipeline: Detection → Classification → Sequence → Translation

COMMANDS:
  python main.py train            → Train YOLOv8 (Stage 1)
  python main.py detect           → Detect signs in image
  python main.py prepare-crops    → Extract crops for Stage 2
  python main.py train-clf        → Train EfficientNet (Stage 2)
  python main.py classify         → Stage 1 + Stage 2
  python main.py sequence         → Stage 1 + 2 + 3
  python main.py translate        → Full pipeline (Stage 1 → 4)
"""


import argparse
import sys
import os


def load_config():
    import yaml
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path  = os.path.join(project_root, "config", "config.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] config.yaml not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════════════

def cmd_train(args):
    from stage1_detection.train import run_training
    run_training(resume=args.resume)


def cmd_detect(args):
    from stage1_detection.detect import run_detection, visualize_detections
    cfg = load_config()
    detections = run_detection(
        image_path   = args.image,
        weights_path = args.weights or cfg["paths"]["stage1_weights"],
        conf         = args.conf    or cfg["detection"]["conf_threshold"],
        save         = True,
        output_dir   = os.path.join(cfg["detection"]["output_dir"], "inference"),
    )
    if args.visualize:
        visualize_detections(args.image, detections)
    return detections


# ══════════════════════════════════════════════════════════════
# STAGE 2
# ══════════════════════════════════════════════════════════════

def cmd_prepare_crops(args):
    from stage1_detection.utils        import locate_dataset_root, download_dataset
    from stage2_classification.dataset import extract_crops
    cfg = load_config()
    ds  = cfg["dataset"]
    if args.data_root:
        dataset_root = locate_dataset_root(args.data_root)
    elif ds.get("local_root") and os.path.exists(str(ds["local_root"])):
        dataset_root = locate_dataset_root(ds["local_root"])
    else:
        dataset_root = download_dataset(ds["kaggle_slug"])
    extract_crops(dataset_root=dataset_root, output_dir=cfg["classification"]["crops_dir"])
    print("[prepare-crops] Done!")


def cmd_train_clf(args):
    from stage2_classification.train import run_training
    run_training(resume=args.resume)


def cmd_classify(args):
    from stage1_detection.detect        import run_detection, visualize_detections
    from stage2_classification.classify import HieroglyphClassifierInference
    cfg = load_config()

    print("\n[Pipeline] Stage 1: Symbol Detection")
    detections = run_detection(
        image_path   = args.image,
        weights_path = args.weights1 or cfg["paths"]["stage1_weights"],
        conf         = cfg["detection"]["conf_threshold"],
        save         = args.visualize,
    )

    print("\n[Pipeline] Stage 2: Symbol Classification")
    clf      = HieroglyphClassifierInference(
        weights_path = args.weights2 or cfg["paths"]["stage2_weights"]
    )
    enriched = clf.classify_detections(args.image, detections)

    print("\n" + "=" * 60)
    print(f"{'YOLO':10s} {'EfficientNet':12s} {'Conf':6s} {'Match':5s}")
    print("=" * 60)
    for d in enriched:
        match = "✓" if d["code"] == d["clf_code"] else "≠"
        print(f"  {d['code']:8s}  {d['clf_code']:10s}  {d['clf_conf']:.2%}  {match}")

    if args.visualize:
        visualize_detections(args.image, enriched)
    return enriched


# ══════════════════════════════════════════════════════════════
# STAGE 3
# ══════════════════════════════════════════════════════════════

def cmd_sequence(args):
    from stage1_detection.detect        import run_detection, visualize_detections
    from stage2_classification.classify import HieroglyphClassifierInference
    from stage3_sequence.sequence       import Stage3Pipeline
    cfg = load_config()

    print("\n[Pipeline] Stage 1: Symbol Detection")
    detections = run_detection(
        image_path   = args.image,
        weights_path = args.weights1 or cfg["paths"]["stage1_weights"],
        conf         = cfg["detection"]["conf_threshold"],
        save         = False,
    )

    print("\n[Pipeline] Stage 2: Symbol Classification")
    clf      = HieroglyphClassifierInference(
        weights_path = args.weights2 or cfg["paths"]["stage2_weights"]
    )
    enriched = clf.classify_detections(args.image, detections)

    print("\n[Pipeline] Stage 3: Sequence Processing")
    pipeline = Stage3Pipeline(image_path=args.image)
    seq      = pipeline.process(enriched)

    if args.save_json:
        out = os.path.join(cfg["sequence"]["output_dir"], "sequence_output.json")
        pipeline.save(seq, out)

    if args.visualize:
        visualize_detections(args.image, enriched)
    return seq


# ══════════════════════════════════════════════════════════════
# STAGE 4 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def cmd_translate(args):
    from stage1_detection.detect        import run_detection, visualize_detections
    from stage2_classification.classify import HieroglyphClassifierInference
    from stage3_sequence.sequence       import Stage3Pipeline
    from stage4_translation.translator  import Stage4Pipeline
    cfg = load_config()

    print("\n[Pipeline] Stage 1: Symbol Detection")
    detections = run_detection(
        image_path   = args.image,
        weights_path = args.weights1 or cfg["paths"]["stage1_weights"],
        conf         = cfg["detection"]["conf_threshold"],
        save         = args.visualize,
    )

    print("\n[Pipeline] Stage 2: Symbol Classification")
    clf      = HieroglyphClassifierInference(
        weights_path = args.weights2 or cfg["paths"]["stage2_weights"]
    )
    enriched = clf.classify_detections(args.image, detections)

    print("\n[Pipeline] Stage 3: Sequence Processing")
    seq_pipe = Stage3Pipeline(image_path=args.image)
    seq      = seq_pipe.process(enriched)

    print("\n[Pipeline] Stage 4: Translation (Groq + Llama 3)")
    tr_pipe = Stage4Pipeline()
    result  = tr_pipe.translate(seq)

    if args.save:
        tr_pipe.save(result)

    if args.visualize:
        visualize_detections(args.image, enriched)
    return result


# ══════════════════════════════════════════════════════════════
# PARSER
# ══════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="Hieroglyph Recognition & Translation System"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Train YOLOv8 (Stage 1)")
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("detect", help="Detect signs in image")
    p.add_argument("--image",     required=True)
    p.add_argument("--weights",   default=None)
    p.add_argument("--conf",      type=float, default=None)
    p.add_argument("--visualize", action="store_true")

    p = sub.add_parser("prepare-crops", help="Extract crops (Stage 2 prep)")
    p.add_argument("--data-root", default=None)

    p = sub.add_parser("train-clf", help="Train EfficientNet (Stage 2)")
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("classify", help="Stage 1 + Stage 2")
    p.add_argument("--image",     required=True)
    p.add_argument("--weights1",  default=None)
    p.add_argument("--weights2",  default=None)
    p.add_argument("--visualize", action="store_true")

    p = sub.add_parser("sequence", help="Stage 1 + 2 + 3")
    p.add_argument("--image",     required=True)
    p.add_argument("--weights1",  default=None)
    p.add_argument("--weights2",  default=None)
    p.add_argument("--save-json", action="store_true", dest="save_json")
    p.add_argument("--visualize", action="store_true")

    p = sub.add_parser("translate", help="Full pipeline: Stage 1 → 4")
    p.add_argument("--image",     required=True)
    p.add_argument("--weights1",  default=None)
    p.add_argument("--weights2",  default=None)
    p.add_argument("--save",      action="store_true")
    p.add_argument("--visualize", action="store_true")

    return parser


COMMAND_MAP = {
    "train"         : cmd_train,
    "detect"        : cmd_detect,
    "prepare-crops" : cmd_prepare_crops,
    "train-clf"     : cmd_train_clf,
    "classify"      : cmd_classify,
    "sequence"      : cmd_sequence,
    "translate"     : cmd_translate,
}

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    COMMAND_MAP[args.command](args)