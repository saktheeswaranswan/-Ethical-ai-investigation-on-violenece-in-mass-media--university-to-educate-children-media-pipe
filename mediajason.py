import os
import cv2
import json
import math
import numpy as np
import mediapipe as mp
from collections import OrderedDict

# =================== CONFIG ===================
VIDEO_IN = "hookayuda.mp4"      # input video file or 0 for webcam
OUTPUT_VIDEO = "output/annotated_mediapipe.mp4"
OUTPUT_JSON = "output/pose_data_mediapipe.json"
WEIGHT_KG = 60.0                # person's mass in kg
G = 9.81                        # gravity constant

# Drawing parameters
SCALE_PIXELS_PER_N = 0.025
TAIL_LEN_PX = 28
BASE_ARROW_THICKNESS = 3
BASE_TAIL_THICKNESS = 6
ARC_RADIUS_PX = 45
ARC_THICKNESS = 3
ELLIPSE_BASE_A = 38
ELLIPSE_BASE_B = 22
ELLIPSE_MAG_SCALE = 0.12
MAG_TEXT_COLOR = (0, 140, 255)
LEG_COLOR = (255, 255, 255)
SKELETON_COLOR = (200, 200, 200)
ARROW_COLOR = (0, 0, 255)
TAIL_COLOR = (0, 255, 255)
ARC_COLOR = (0, 255, 0)
JOINT_TEXT_COLOR = (255, 200, 0)  # joint angle text color
# ==============================================

os.makedirs("output", exist_ok=True)
mp_pose = mp.solutions.pose

# ---------- Utility helpers ----------
def safe_point(landmark, w, h):
    return np.array([float(landmark.x * w), float(landmark.y * h)])

def compute_angle(a, b, c):
    """
    Angle at b formed by a-b-c (in degrees).
    a,b,c are numpy arrays of coordinates.
    Returns angle in degrees or None if degenerate.
    """
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    return ang

def draw_text_bg(img, text, pos, font_scale=0.5, color=(255,255,255), bg_color=(0,0,0), thickness=1):
    """Draw text with small background rectangle for readability."""
    x, y = int(pos[0]), int(pos[1])
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x-3, y-h-3), (x + w + 3, y + 3), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_force_arrow(img, origin, vec, magnitude_N,
                     scale_pixels_per_N=SCALE_PIXELS_PER_N,
                     tail_len_px=TAIL_LEN_PX,
                     base_arrow_thickness=BASE_ARROW_THICKNESS,
                     base_tail_thickness=BASE_TAIL_THICKNESS,
                     arrow_color=ARROW_COLOR,
                     tail_color=TAIL_COLOR):
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    vec_unit = vec / norm
    length_px = float(magnitude_N) * scale_pixels_per_N
    arrow_th = int(max(1, base_arrow_thickness + length_px / 40))
    tail_th = int(max(1, base_tail_thickness + length_px / 60))
    tail_end = origin + vec_unit * tail_len_px
    cv2.line(img, tuple(origin.astype(int)), tuple(tail_end.astype(int)), tail_color, tail_th)
    end = origin + vec_unit * length_px
    cv2.arrowedLine(img, tuple(origin.astype(int)), tuple(end.astype(int)), arrow_color, arrow_th, tipLength=0.28)
    # magnitude label
    perp = np.array([-vec_unit[1], vec_unit[0]])
    label_pos = end + perp * 10 + vec_unit * 6
    draw_text_bg(img, f"{magnitude_N:.1f} N", (int(label_pos[0]), int(label_pos[1])), font_scale=0.55, color=MAG_TEXT_COLOR, bg_color=(20,20,20), thickness=2)
    return {
        "origin": [float(origin[0]), float(origin[1])],
        "end": [float(end[0]), float(end[1])],
        "tail_end": [float(tail_end[0]), float(tail_end[1])],
        "unit_vec": [float(vec_unit[0]), float(vec_unit[1])],
        "length_px": float(length_px),
        "arrow_thickness_px": arrow_th,
        "tail_thickness_px": tail_th
    }

def draw_angle_arc(img, center, leg_unit, down_unit=np.array([0.0, 1.0]), radius_px=ARC_RADIUS_PX, color=ARC_COLOR, thickness=ARC_THICKNESS):
    def norm360(a):
        a = a % 360.0
        return a + 360.0 if a < 0 else a
    ang_leg = norm360(math.degrees(math.atan2(leg_unit[1], leg_unit[0])))
    ang_down = norm360(math.degrees(math.atan2(down_unit[1], down_unit[0])))
    diff = (ang_leg - ang_down + 360.0) % 360.0
    if diff > 180.0:
        start_angle, end_angle = ang_leg, ang_down
    else:
        start_angle, end_angle = ang_down, ang_leg
    center_int = (int(center[0]), int(center[1]))
    cv2.ellipse(img, center_int, (radius_px, radius_px), 0.0, start_angle, end_angle, color, thickness)
    end_rad = math.radians(end_angle)
    arrow_pt = (int(center[0] + radius_px * math.cos(end_rad)), int(center[1] + radius_px * math.sin(end_rad)))
    cv2.circle(img, arrow_pt, 5, color, -1)
    included = abs((ang_leg - ang_down + 360.0) % 360.0)
    if included > 180.0:
        included = 360.0 - included
    return {"center":[float(center[0]), float(center[1])], "start_angle_deg":float(start_angle), "end_angle_deg":float(end_angle), "included_angle_deg":float(included), "arc_end_point":[float(arrow_pt[0]), float(arrow_pt[1])]}

def draw_variable_ellipse(img, center, leg_unit, length_px, base_a=ELLIPSE_BASE_A, base_b=ELLIPSE_BASE_B, mag_scale=ELLIPSE_MAG_SCALE, color=(120,180,60), thickness=2):
    a = int(round(base_a + length_px * mag_scale))
    b = int(round(base_b + length_px * mag_scale * 0.55))
    leg_angle_deg = math.degrees(math.atan2(leg_unit[1], leg_unit[0]))
    center_int = (int(center[0]), int(center[1]))
    cv2.ellipse(img, center_int, (a, b), leg_angle_deg, 0, 360, color, thickness)
    return {"center":[float(center[0]), float(center[1])], "semi_axis_a_px":float(a), "semi_axis_b_px":float(b), "rotation_deg":float(leg_angle_deg)}

# ---------- Main processing ----------
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

pose_connections = [(mp_pose.PoseLandmark(a).name, mp_pose.PoseLandmark(b).name) for a,b in mp_pose.POSE_CONNECTIONS]
pose_data = []
frame_idx = 0
weight_N = WEIGHT_KG * G
down = np.array([0.0, 1.0])

# Define joint triplets (a, b, c) -> angle at b using a-b-c
JOINT_DEFINITIONS = {
    # Left side
    "LEFT_ELBOW": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "LEFT_SHOULDER": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "LEFT_WRIST": ("LEFT_ELBOW", "LEFT_WRIST", "LEFT_INDEX"),  # wrist uses index finger if visible
    "LEFT_KNEE": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "LEFT_HIP": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "LEFT_ANKLE": ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    # Right side
    "RIGHT_ELBOW": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "RIGHT_SHOULDER": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "RIGHT_WRIST": ("RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"),
    "RIGHT_KNEE": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "RIGHT_HIP": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "RIGHT_ANKLE": ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
}

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated = frame.copy()

        frame_record = OrderedDict([
            ("frame", int(frame_idx)),
            ("time_s", float(frame_idx / fps)),
            ("keypoints", {}),
            ("edges", pose_connections),
            ("foot_force", {}),
            ("joint_angles", {}),
            ("live_inference", {})
        ])

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # collect keypoints
            pts = {}
            for name in mp_pose.PoseLandmark:
                p = lm[name.value]
                pts[name.name] = {"x": float(p.x * width), "y": float(p.y * height), "visibility": float(getattr(p, "visibility", 1.0))}
            frame_record["keypoints"] = pts

            # draw skeleton & landmarks
            for a_name, b_name in pose_connections:
                pa, pb = pts[a_name], pts[b_name]
                if pa["visibility"] > 0.1 and pb["visibility"] > 0.1:
                    cv2.line(annotated, (int(pa["x"]), int(pa["y"])), (int(pb["x"]), int(pb["y"])), SKELETON_COLOR, 2)
            for nm, p in pts.items():
                if p["visibility"] > 0.1:
                    cv2.circle(annotated, (int(p["x"]), int(p["y"])), 3, LEG_COLOR, -1)

            # compute foot forces & ellipses (keeps earlier functionality)
            for side in ["LEFT", "RIGHT"]:
                hip_key = f"{side}_HIP"
                ankle_key = f"{side}_ANKLE"
                if hip_key not in pts or ankle_key not in pts:
                    continue
                hip = np.array([pts[hip_key]["x"], pts[hip_key]["y"]])
                ankle = np.array([pts[ankle_key]["x"], pts[ankle_key]["y"]])
                if np.linalg.norm(hip) < 1e-3 or np.linalg.norm(ankle) < 1e-3:
                    continue
                leg_vec = ankle - hip
                if np.linalg.norm(leg_vec) < 1e-6:
                    continue
                leg_unit = leg_vec / np.linalg.norm(leg_vec)
                cos_theta = np.dot(leg_unit, down) / (np.linalg.norm(down) * np.linalg.norm(leg_unit))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta_rad = math.acos(cos_theta)
                theta_deg = math.degrees(theta_rad)
                F_perp = weight_N * abs(math.sin(theta_rad))
                perp = np.array([-leg_unit[1], leg_unit[0]])
                perp_unit = perp / np.linalg.norm(perp)
                if np.dot(down, perp_unit) < 0:
                    perp_unit = -perp_unit

                # draw arrow, ellipse, arc and label angle
                arrow_info = draw_force_arrow(annotated, ankle, perp_unit, F_perp)
                ellipse_info = draw_variable_ellipse(annotated, ankle, leg_unit, arrow_info["length_px"] if arrow_info else 0.0)
                arc_info = draw_angle_arc(annotated, ankle, leg_unit, down_unit=down)
                draw_text_bg(annotated, f"{theta_deg:.1f}°", (int(ankle[0]+8), int(ankle[1] - ARC_RADIUS_PX - 8)), font_scale=0.6, color=ARC_COLOR, bg_color=(10,10,10), thickness=2)

                frame_record["foot_force"][side.lower()] = {
                    "origin_pixel": [float(ankle[0]), float(ankle[1])],
                    "hip_pixel": [float(hip[0]), float(hip[1])],
                    "leg_angle_from_vertical_deg": float(theta_deg),
                    "perp_force_N": float(F_perp),
                    "perp_unit_vector": [float(perp_unit[0]), float(perp_unit[1])],
                    "arrow_draw": arrow_info,
                    "arc_info": arc_info,
                    "ellipse_info": ellipse_info
                }

                frame_record["live_inference"][f"{side}_angle_deg"] = round(theta_deg, 2)
                frame_record["live_inference"][f"{side}_F_perp_N"] = round(F_perp, 2)

            # --- NEW: compute & draw all joint angles defined above ---
            for joint_name, (a_name, b_name, c_name) in JOINT_DEFINITIONS.items():
                # ensure landmarks exist and visible
                pa = pts.get(a_name)
                pb = pts.get(b_name)
                pc = pts.get(c_name)
                if pa is None or pb is None or pc is None:
                    # skip if any missing
                    continue
                if pa["visibility"] < 0.1 or pb["visibility"] < 0.1 or pc["visibility"] < 0.1:
                    continue

                A = np.array([pa["x"], pa["y"]])
                B = np.array([pb["x"], pb["y"]])
                C = np.array([pc["x"], pc["y"]])

                angle = compute_angle(A, B, C)
                if angle is None:
                    continue

                # draw angle text near joint B
                text_pos = (int(B[0] + 6), int(B[1] - 6))
                draw_text_bg(annotated, f"{angle:.1f}°", text_pos, font_scale=0.5, color=JOINT_TEXT_COLOR, bg_color=(10,10,10), thickness=1)

                # optionally draw small arc representing the angle between BA and BC
                # compute directions for arc
                v1 = A - B
                v2 = C - B
                a1 = math.degrees(math.atan2(v1[1], v1[0]))
                a2 = math.degrees(math.atan2(v2[1], v2[0]))
                # draw small arc radius based on body scale
                arc_r = 20
                # normalize angles for cv2.ellipse usage
                start_ang = a1 % 360
                end_ang = a2 % 360
                # choose direction to draw shorter arc
                diff = (end_ang - start_ang + 360) % 360
                if diff > 180:
                    # swap so small arc drawn
                    start_angle_for_cv = end_ang
                    end_angle_for_cv = start_ang
                else:
                    start_angle_for_cv = start_ang
                    end_angle_for_cv = end_ang
                try:
                    cv2.ellipse(annotated, (int(B[0]), int(B[1])), (arc_r, arc_r), 0.0, start_angle_for_cv, end_angle_for_cv, JOINT_TEXT_COLOR, 1)
                except Exception:
                    pass  # if ellipse fails for degenerate angles, ignore

                # save to JSON
                frame_record["joint_angles"][joint_name] = float(round(angle, 3))

        else:
            cv2.putText(annotated, "No pose detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # frame counter
        draw_text_bg(annotated, f"Frame: {frame_idx}", (12, height - 10), font_scale=0.6, color=(230,230,230), bg_color=(10,10,10), thickness=1)

        out.write(annotated)
        pose_data.append(frame_record)
        frame_idx += 1

cap.release()
out.release()

# save JSON with joint angles + all previous info
with open(OUTPUT_JSON, "w") as f:
    json.dump(pose_data, f, indent=2)

print("✅ Done.")
print(f"Annotated video: {OUTPUT_VIDEO}")
print(f"Pose + joint angles JSON: {OUTPUT_JSON}")

