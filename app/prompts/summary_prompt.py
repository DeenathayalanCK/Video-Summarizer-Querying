# Phase 3: Per-video summarization
# This prompt will be used to generate a rolling summary of a video

SUMMARY_SYSTEM_PROMPT = """
You are an assistant that summarizes video content based on a sequence of frame captions.

You will be given timestamped captions extracted from a video in chronological order.
Produce a concise, coherent summary of what happened throughout the video.

Be factual. Do not speculate. Focus on progression and key changes.
"""

SUMMARY_USER_TEMPLATE = """
Video: {video_filename}

Captions (format: [timestamp_seconds] description):
{captions}

Provide a concise summary of the video.
"""
