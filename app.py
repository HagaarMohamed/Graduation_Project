import os
import tempfile
from pathlib import Path

import streamlit as st

from main import load_config
from stage1_detection.detect import run_detection
from stage2_classification.classify import HieroglyphClassifierInference
from stage3_sequence.sequence import Stage3Pipeline
from stage4_translation.translator import Stage4Pipeline


st.set_page_config(
    page_title="Civilization Ambassador",
    page_icon="🏺",
    layout="wide",
)


# Convert custom pipeline objects into values Streamlit can display.
def displayable(value, depth=0):
    if depth > 5:
        return "..."

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): displayable(item, depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [displayable(item, depth + 1) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return displayable(value.tolist(), depth + 1)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        return {
            str(key): displayable(item, depth + 1)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }

    return str(value)


def show_stage_result(title, result):
    """Show the actual result of a stage inside an expandable panel."""
    with st.expander(title, expanded=True):
        converted = displayable(result)
        if isinstance(converted, (dict, list)):
            st.json(converted)
        else:
            st.write(converted)


st.title("Civilization Ambassador")
st.subheader("نظام اكتشاف وتصنيف وترجمة النقوش الهيروغليفية")

left_column, right_column = st.columns([1, 1])

with left_column:
    st.markdown("### الصورة المرفوعة")
    uploaded_file = st.file_uploader(
        "اختر صورة النقش الهيروغليفي",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption="الصورة المدخلة",
            use_container_width=True,
        )

with right_column:
    st.markdown("### نتائج المراحل الأربع")

    if uploaded_file is None:
        st.info("ارفع صورة هيروغليفية أولًا لبدء التحليل.")
    elif st.button("Discover with ANKH", type="primary", use_container_width=True):
        temporary_path = None
        progress = st.progress(0, text="Preparing the image...")

        try:
            file_suffix = Path(uploaded_file.name or "image.jpg").suffix.lower()
            if file_suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                file_suffix = ".jpg"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_suffix,
            ) as temporary_file:
                temporary_file.write(uploaded_file.getbuffer())
                temporary_path = temporary_file.name

            config = load_config()

            # -----------------------------------------------------------------
            # Stage 1: Detection
            # -----------------------------------------------------------------
            with st.status("Stage 1 — Detection", expanded=True) as stage_status:
                st.write("Detecting hieroglyphic signs...")
                detections = run_detection(
                    image_path=temporary_path,
                    weights_path=config["paths"]["stage1_weights"],
                    conf=config["detection"]["conf_threshold"],
                    save=False,
                )
                stage_status.update(
                    label="Stage 1 — Detection completed",
                    state="complete",
                )
            show_stage_result("Stage 1 result — Detected signs", detections)
            progress.progress(25, text="Stage 1 completed")

            # -----------------------------------------------------------------
            # Stage 2: Classification
            # -----------------------------------------------------------------
            with st.status("Stage 2 — Classification", expanded=True) as stage_status:
                st.write("Classifying the detected signs...")
                classifier = HieroglyphClassifierInference(
                    weights_path=config["paths"]["stage2_weights"]
                )
                enriched_detections = classifier.classify_detections(
                    temporary_path,
                    detections,
                )
                stage_status.update(
                    label="Stage 2 — Classification completed",
                    state="complete",
                )
            show_stage_result(
                "Stage 2 result — Classified signs",
                enriched_detections,
            )
            progress.progress(50, text="Stage 2 completed")

            # -----------------------------------------------------------------
            # Stage 3: Sequence Processing
            # -----------------------------------------------------------------
            with st.status(
                "Stage 3 — Sequence Processing",
                expanded=True,
            ) as stage_status:
                st.write("Building the ordered hieroglyphic sequence...")
                sequence_pipeline = Stage3Pipeline(image_path=temporary_path)
                sequence = sequence_pipeline.process(enriched_detections)
                stage_status.update(
                    label="Stage 3 — Sequence Processing completed",
                    state="complete",
                )
            show_stage_result("Stage 3 result — Ordered sequence", sequence)
            progress.progress(75, text="Stage 3 completed")

            # -----------------------------------------------------------------
            # Stage 4: Translation
            # -----------------------------------------------------------------
            with st.status("Stage 4 — Translation", expanded=True) as stage_status:
                st.write("Sending the image and sequence to the Vision translator...")
                # The image path is required so Qwen Vision receives the actual image.
                translator = Stage4Pipeline(image_path=temporary_path)
                result = translator.translate(sequence)
                stage_status.update(
                    label="Stage 4 — Translation completed",
                    state="complete",
                )
            progress.progress(100, text="All four stages completed")

            show_stage_result(
                "Stage 4 result — Translation",
                {
                    "english_translation": result.english_translation,
                    "arabic_translation": result.arabic_translation,
                    "confidence": result.confidence,
                },
            )

            st.success("Analysis completed successfully")

            st.markdown("## Translation in English")
            st.write(result.english_translation or "No English translation returned.")

            st.markdown("## الترجمة العربية")
            st.write(result.arabic_translation or "لم يتم إرجاع ترجمة عربية.")

            st.markdown("## Confidence")
            st.write(result.confidence or "medium")

        except Exception as error:
            st.error(f"حدث خطأ أثناء التحليل: {error}")

        finally:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
