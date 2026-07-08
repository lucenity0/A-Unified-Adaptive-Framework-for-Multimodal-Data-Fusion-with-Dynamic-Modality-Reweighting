"""
Small helpers for extracting projected CLIP embeddings.

Some Transformers/architecture combinations return model-output objects where
older code expected tensors from get_text_features/get_image_features. Calling
the encoders and projection layers directly keeps the feature shape stable.
"""


def extract_clip_text_features(clip, input_ids, attention_mask):
    text_out = clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
    return clip.text_projection(text_out.pooler_output)


def extract_clip_image_features(clip, pixel_values):
    image_out = clip.vision_model(pixel_values=pixel_values)
    return clip.visual_projection(image_out.pooler_output)
