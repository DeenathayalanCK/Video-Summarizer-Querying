"""
memory_graph.py — Semantic Video Memory Graph.

Instead of storing only embeddings (which answer "find similar text"),
the Memory Graph stores structured semantic FACTS extracted from the video:

  Node types:
    identity    — "Person #3, male adult, white shirt, SECURITY badge"
    behaviour   — "Person #3 was loitering from 00:12 to 01:04"
    motion      — "Person #3 was running at 00:32, sudden_stop at 00:45"
    scene       — "crowd_gathering at 00:05–00:45 involving tracks [3,5,7]"
    relationship— "Person #3 and Car #1 were present simultaneously 00:05–00:07"
    timeline    — "At 00:09 Person #3 stopped abruptly"

  Each node has:
    - node_type      : identity | behaviour | motion | scene | relationship | timeline
    - node_label     : short human-readable name ("Person #3 loitering")
    - semantic_text  : full natural-language fact (used in QA context)
    - track_id       : which track (None for scene/relationship nodes)
    - start_second   : when this fact starts
    - end_second     : when it ends
    - confidence     : 0–1
    - metadata       : JSONB for any extra structured fields

One SemanticMemoryGraph row per video.  Nodes stored as JSONB array.
Built by MemoryGraphBuilder after TimelineBuilder runs.

The QA engine reads graph nodes for a video and injects them as
Layer 1 context — the most semantic, pre-digested knowledge available.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List
from app.core.logging import get_logger


@dataclass
class MemoryNode:
    node_type: str        # identity | behaviour | motion | scene | relationship | timeline
    node_label: str       # "Person #3 loitering"
    semantic_text: str    # full natural-language fact
    track_id: Optional[int]
    start_second: float
    end_second: float
    confidence: float
    node_meta: dict

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryGraphBuilder:
    """
    Builds a SemanticMemoryGraph for a video from TrackEvent data.

    Input:  TrackEvent rows (with attributes["temporal"] and attributes["motion_summary"])
            + VideoTimeline (for scene events)
    Output: SemanticMemoryGraph row with nodes[] JSONB array

    Called by VideoIntelligenceProcessor after TimelineBuilder.
    Also callable manually via POST /build-memory-graph/{video}.
    """

    def __init__(self, db):
        self.db = db
        self.logger = get_logger()

    def build(self, video_filename: str, camera_id: str):
        """Build and persist the memory graph. Returns SemanticMemoryGraph object."""
        from app.storage.models import TrackEvent, VideoTimeline, SemanticMemoryGraph

        entry_events = (
            self.db.query(TrackEvent)
            .filter(
                TrackEvent.video_filename == video_filename,
                TrackEvent.event_type == "entry",
            )
            .order_by(TrackEvent.first_seen_second)
            .all()
        )

        if not entry_events:
            self.logger.warning("memory_graph_no_events", video=video_filename)
            return None

        nodes: List[MemoryNode] = []

        for ev in entry_events:
            attrs  = ev.attributes or {}
            temp   = attrs.get("temporal", {})
            ms     = attrs.get("motion_summary", {})
            tid    = ev.track_id
            obj    = ev.object_class
            t_in   = ev.first_seen_second
            t_out  = ev.last_seen_second
            dur    = ev.duration_seconds

            # ── Identity node ─────────────────────────────────────────────────
            id_parts = self._identity_parts(obj, attrs)
            id_label = f"{obj.capitalize()} #{tid}"
            id_text  = f"{id_label}: {', '.join(id_parts)}. " \
                       f"First seen {self._fmt(t_in)}, last seen {self._fmt(t_out)}, " \
                       f"duration {dur:.0f}s, confidence {ev.best_confidence or 0:.0%}."
            nodes.append(MemoryNode(
                node_type="identity",
                node_label=id_label,
                semantic_text=id_text,
                track_id=tid,
                start_second=t_in,
                end_second=t_out,
                confidence=ev.best_confidence or 0.5,
                metadata={"object_class": obj, "attributes": id_parts},
            ))

            # ── Plate number node (vehicles only) ────────────────────────────
            if not (obj in ("car","truck","bus","motorcycle","bicycle")):
                pass  # handled below
            else:
                plate_num = attrs.get("plate_number", "unknown")
                if plate_num and plate_num not in ("unknown", ""):
                    nodes.append(MemoryNode(
                        node_type="identity",
                        node_label=f"{id_label} plate={plate_num}",
                        semantic_text=(
                            f"LICENSE PLATE DETECTED — {id_label} has plate number: {plate_num}. "
                            f"Vehicle first seen {self._fmt(t_in)}, last seen {self._fmt(t_out)}. "
                            f"This plate was read by the vision model from the best crop."
                        ),
                        track_id=tid,
                        start_second=t_in,
                        end_second=t_out,
                        confidence=0.75,  # OCR from low-res crop — moderate confidence
                        metadata={"plate_number": plate_num, "object_class": obj},
                    ))

            # ── Behaviour node ────────────────────────────────────────────────
            behaviour = temp.get("behaviour", "")
            if behaviour and behaviour != "unknown":
                notes = temp.get("notes", "")
                appc  = temp.get("appearance_count", 1)
                mvmt  = temp.get("movement_pattern", "")
                b_text = (
                    f"{id_label} behaviour: {behaviour}. "
                    f"{notes} "
                    f"({'appeared ' + str(appc) + ' times' if appc > 1 else 'single continuous visit'})"
                    f"{', movement: ' + mvmt if mvmt else ''}."
                )
                nodes.append(MemoryNode(
                    node_type="behaviour",
                    node_label=f"{id_label} {behaviour}",
                    semantic_text=b_text,
                    track_id=tid,
                    start_second=t_in,
                    end_second=t_out,
                    confidence=0.85,
                    metadata={"behaviour": behaviour, "appearance_count": appc},
                ))

            # ── Motion node ───────────────────────────────────────────────────
            dom_state  = ms.get("dominant_state", "")
            mevts      = ms.get("motion_events", [])
            direction  = ms.get("direction", "")

            if dom_state and dom_state not in ("stationary", "unknown"):
                m_parts = [f"predominantly {dom_state}"]
                if direction and direction != "stationary":
                    m_parts.append(f"moving {direction.replace('_',' ')}")
                if mevts:
                    m_parts.append(f"events: {'; '.join(mevts[:3])}")
                m_text = f"{id_label} motion: {', '.join(m_parts)}."
                nodes.append(MemoryNode(
                    node_type="motion",
                    node_label=f"{id_label} {dom_state}",
                    semantic_text=m_text,
                    track_id=tid,
                    start_second=t_in,
                    end_second=t_out,
                    confidence=0.80,
                    metadata={"dominant_state": dom_state, "motion_events": mevts},
                ))

            # ── Per motion-event timeline nodes ───────────────────────────────
            for mev in mevts:
                try:
                    sec = float(mev.split("@")[1].strip().split("s")[0])
                except (IndexError, ValueError):
                    sec = t_in

                if "fall_proxy" in mev:
                    label = f"FALL ALERT: {id_label}"
                    text  = (
                        f"SAFETY ALERT — {id_label} may have fallen at {self._fmt(sec)}. "
                        f"Bounding box changed from upright to horizontal. "
                        f"Raw signal: {mev}. Verify footage immediately."
                    )
                    conf = 0.70
                elif "sudden_stop" in mev:
                    label = f"{id_label} sudden stop"
                    text  = f"{id_label} stopped abruptly at {self._fmt(sec)}. May indicate object interaction, confrontation, or surprise. {mev}."
                    conf = 0.75
                elif "direction_change" in mev:
                    label = f"{id_label} direction change"
                    text  = f"{id_label} reversed direction at {self._fmt(sec)}. May indicate hesitation or awareness of surroundings."
                    conf = 0.65
                else:
                    continue

                nodes.append(MemoryNode(
                    node_type="timeline",
                    node_label=label,
                    semantic_text=text,
                    track_id=tid,
                    start_second=sec,
                    end_second=sec,
                    confidence=conf,
                    metadata={"motion_event": mev},
                ))

        # ── Scene-level nodes from VideoTimeline ──────────────────────────────
        tl = (
            self.db.query(VideoTimeline)
            .filter(VideoTimeline.video_filename == video_filename)
            .first()
        )
        if tl:
            for se in (tl.scene_events or []):
                et  = se.get("event_type", "")
                s   = se.get("start_second", 0)
                e   = se.get("end_second", 0)
                tids = se.get("track_ids", [])
                nodes.append(MemoryNode(
                    node_type="scene",
                    node_label=f"Scene: {et}",
                    semantic_text=(
                        f"SCENE EVENT — {et.replace('_',' ')} detected "
                        f"from {self._fmt(s)} to {self._fmt(e)} "
                        f"involving tracks {tids}. {se.get('notes','')}"
                    ),
                    track_id=None,
                    start_second=s,
                    end_second=e,
                    confidence=se.get("confidence", 0.5),
                    metadata={"event_type": et, "track_ids": tids},
                ))

        # ── Co-presence relationship nodes ────────────────────────────────────
        # Any two tracks that overlap in time → co-presence fact
        ev_list = list(entry_events)
        for i in range(len(ev_list)):
            for j in range(i + 1, len(ev_list)):
                a, b = ev_list[i], ev_list[j]
                overlap_start = max(a.first_seen_second, b.first_seen_second)
                overlap_end   = min(a.last_seen_second,  b.last_seen_second)
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    nodes.append(MemoryNode(
                        node_type="relationship",
                        node_label=f"{a.object_class} #{a.track_id} + {b.object_class} #{b.track_id}",
                        semantic_text=(
                            f"{a.object_class.capitalize()} #{a.track_id} and "
                            f"{b.object_class} #{b.track_id} were present simultaneously "
                            f"from {self._fmt(overlap_start)} to {self._fmt(overlap_end)} "
                            f"({overlap:.0f}s overlap)."
                        ),
                        track_id=None,
                        start_second=overlap_start,
                        end_second=overlap_end,
                        confidence=0.95,
                        metadata={
                            "track_a": a.track_id, "class_a": a.object_class,
                            "track_b": b.track_id, "class_b": b.object_class,
                            "overlap_seconds": round(overlap, 1),
                        },
                    ))

        # Limit relationship nodes to avoid explosion on crowded videos
        rel_nodes   = [n for n in nodes if n.node_type == "relationship"][:20]
        other_nodes = [n for n in nodes if n.node_type != "relationship"]
        all_nodes   = other_nodes + rel_nodes

        node_dicts = [n.to_dict() for n in all_nodes]

        # ── Persist ───────────────────────────────────────────────────────────
        from app.storage.models import SemanticMemoryGraph as SMG
        existing = (
            self.db.query(SMG)
            .filter(SMG.video_filename == video_filename)
            .all()
        )
        for row in existing:
            self.db.delete(row)
        self.db.flush()

        # Store each node as its own row for efficient querying
        for nd in all_nodes:
            row = SMG(
                video_filename=video_filename,
                camera_id=camera_id,
                node_type=nd.node_type,
                node_label=nd.node_label,
                semantic_text=nd.semantic_text,
                track_id=nd.track_id,
                start_second=nd.start_second,
                end_second=nd.end_second,
                confidence=nd.confidence,
                node_meta=nd.metadata,
            )
            self.db.add(row)

        self.db.commit()
        self.logger.info(
            "memory_graph_built",
            video=video_filename,
            nodes=len(all_nodes),
            types={t: sum(1 for n in all_nodes if n.node_type == t)
                   for t in ["identity","behaviour","motion","scene","relationship","timeline"]},
        )
        return all_nodes

    @staticmethod
    def _fmt(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _identity_parts(obj: str, attrs: dict) -> list:
        parts = []
        if obj == "person":
            for k in ("gender_estimate","age_estimate","clothing_top","clothing_bottom",
                      "head_covering","carrying","visible_text"):
                v = attrs.get(k, "")
                if v and v not in ("unknown","none",""):
                    parts.append(v)
        else:
            for k in ("color","type","make_estimate"):
                v = attrs.get(k, "")
                if v and v not in ("unknown","none",""):
                    parts.append(v)
            plate = attrs.get("plate_number", "")
            if plate and plate not in ("unknown",""):
                parts.append(f"plate:{plate}")
        return parts if parts else [obj]