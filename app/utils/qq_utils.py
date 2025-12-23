"""qq_utils.py

Utilities for parsing OneBot v11 message segments from QQ.

This module focuses on *robust recognition* of QQ official emoticons/stickers.
It supports:
  - `face` (QQ built-in/system faces)
  - `mface` (QQ sticker store / marketplace emoticons)
  - `image` segments converted from `mface`
  - `dice` / `rps` / `poke` (NapCatQQ and some OneBot implementations)

If an emoticon cannot be mapped to a human-readable name, we still keep it as
`[Face:<id>]` instead of silently dropping it.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Face mapping
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# - Different QQ implementations historically used *different* face-id tables.
# - Recent desktop QQ / QQNT ecosystems commonly follow the "system emoji id"
#   table used by the official QQ Bot (OpenAPI) documentation.
# - Some legacy CQ/CoolQ ecosystems use another classic CQ face-id table.
#
# To be maximally compatible, we:
#   1) First try to extract a readable name from `data.raw` if present.
#   2) Then use OFFICIAL mapping (QQ Bot doc table) as the default.
#   3) Finally fall back to a LEGACY mapping (classic CQ) if the id isn't in OFFICIAL.
#
# You can override the mapping preference via env var:
#   QQ_FACE_MAP_MODE = "official" | "legacy" | "auto" (default: "official")
#
# Optional:
#   QQ_FACE_MAPPING_FILE = /path/to/qq_face_mapping.json
#   (JSON: {"id": "name", ...}) will override built-in mappings.


QQ_FACE_MAP_MODE = os.getenv("QQ_FACE_MAP_MODE", "official").strip().lower()

# Optional user-provided mapping file (JSON: {"id": "name", ...}).
# This is useful if you want to extend/override the built-in mapping without
# editing code again.
QQ_FACE_MAPPING_USER: Dict[str, str] = {}
_user_map_path = os.getenv("QQ_FACE_MAPPING_FILE")
if _user_map_path:
    try:
        with open(_user_map_path, "r", encoding="utf-8") as f:
            _obj = json.load(f)
        if isinstance(_obj, dict):
            QQ_FACE_MAPPING_USER = {str(k): str(v) for k, v in _obj.items() if v is not None}
    except Exception:
        # Keep silent: mapping file is optional.
        QQ_FACE_MAPPING_USER = {}


# Legacy mapping (classic CQ/CoolQ-style) for the early QQ built-in faces.
#
# This table is widely used in older CQHTTP/go-cqhttp ecosystems.
# If your environment still follows the classic CQ ids, set:
#   QQ_FACE_MAP_MODE=legacy
#
# NOTE: some ids overlap with the newer official table but map to different
# names. That's why we keep an explicit mode switch.
QQ_FACE_MAPPING_LEGACY: Dict[str, str] = {
    "0": "惊讶",
    "1": "撇嘴",
    "2": "色",
    "3": "发呆",
    "4": "得意",
    "5": "流泪",
    "6": "害羞",
    "7": "闭嘴",
    "8": "睡",
    "9": "大哭",
    "10": "尴尬",
    "11": "发怒",
    "12": "调皮",
    "13": "呲牙",
    "14": "微笑",
    "15": "难过",
    "16": "酷",
    "17": "抓狂",
    "18": "吐",
    "19": "偷笑",
    "20": "可爱",
    "21": "白眼",
    "22": "傲慢",
    "23": "饥饿",
    "24": "困",
    "25": "惊恐",
    "26": "流汗",
    "27": "憨笑",
    "28": "悠闲",
    "29": "奋斗",
    "30": "咒骂",
    "31": "疑问",
    "32": "嘘",
    "33": "晕",
    "34": "折磨",
    "35": "衰",
    "36": "骷髅",
    "37": "敲打",
    "38": "再见",
    "39": "擦汗",
    "40": "抠鼻",
    "41": "鼓掌",
    "42": "糗大了",
    "43": "坏笑",
    "44": "左哼哼",
    "45": "右哼哼",
    "46": "哈欠",
    "47": "鄙视",
    "48": "委屈",
    "49": "快哭了",
    "50": "阴险",
    "51": "亲亲",
    "52": "吓",
    "53": "可怜",
    "54": "菜刀",
    "55": "西瓜",
    "56": "啤酒",
    "57": "篮球",
    "58": "乒乓",
    "59": "咖啡",
    "60": "饭",
    "61": "猪头",
    "62": "玫瑰",
    "63": "凋谢",
    "64": "嘴唇",
    "65": "爱心",
    "66": "心碎",
    "67": "蛋糕",
    "68": "闪电",
    "69": "炸弹",
    "70": "刀",
    "71": "足球",
    "72": "瓢虫",
    "73": "便便",
    "74": "月亮",
    "75": "太阳",
    "76": "礼物",
    "77": "拥抱",
    "78": "强",
    "79": "弱",
    "80": "握手",
    "81": "胜利",
    "82": "抱拳",
    "83": "勾引",
    "84": "拳头",
    "85": "差劲",
    "86": "爱你",
    "87": "NO",
    "88": "OK",
    "89": "爱情",
    "90": "飞吻",
    "91": "跳跳",
    "92": "发抖",
    "93": "怄火",
    "94": "转圈",
    "95": "磕头",
    "96": "回头",
    "97": "跳绳",
    "98": "投降",
    "99": "激动",
    "100": "乱舞",
    "101": "献吻",
    "102": "左太极",
    "103": "右太极",
}


# "Official" system emoji mapping (EmojiType=1) from QQ Bot OpenAPI docs.
# Note: the official docs themselves state the list is partial and may change
# over time.
QQ_FACE_MAPPING_OFFICIAL: Dict[str, str] = {
    # Basic
    "4": "得意",
    "5": "流泪",
    "8": "睡",
    "9": "大哭",
    "10": "尴尬",
    "12": "调皮",
    "14": "微笑",
    "16": "酷",
    "21": "可爱",
    "23": "傲慢",
    "24": "饥饿",
    "25": "困",
    "26": "惊恐",
    "27": "流汗",
    "28": "憨笑",
    "29": "悠闲",
    "30": "奋斗",
    "32": "疑问",
    "33": "嘘",
    "34": "晕",
    "38": "敲打",
    "39": "再见",
    "41": "发抖",
    "42": "爱情",
    "43": "跳跳",
    "49": "拥抱",
    "53": "蛋糕",
    "60": "咖啡",
    "63": "玫瑰",
    "66": "爱心",
    "74": "太阳",
    "75": "月亮",
    "76": "赞",
    "78": "握手",
    "79": "胜利",
    "85": "飞吻",
    "89": "西瓜",
    "96": "冷汗",
    "97": "擦汗",
    "98": "抠鼻",
    "99": "鼓掌",
    "100": "糗大了",
    "101": "坏笑",
    "102": "左哼哼",
    "103": "右哼哼",
    "104": "哈欠",
    "106": "委屈",
    "109": "左亲亲",
    "111": "可怜",
    "116": "示爱",
    "118": "抱拳",
    "120": "拳头",
    "122": "爱你",
    "123": "NO",
    "124": "OK",
    "125": "转圈",
    "129": "挥手",
    "144": "喝彩",
    "147": "棒棒糖",
    # Newer system emojis
    "171": "茶",
    "173": "泪奔",
    "174": "无奈",
    "175": "卖萌",
    "176": "小纠结",
    "179": "doge",
    "180": "惊喜",
    "181": "骚扰",
    "182": "笑哭",
    "183": "我最美",
    "201": "点赞",
    "203": "托脸",
    "212": "托腮",
    "214": "啵啵",
    "219": "蹭一蹭",
    "222": "抱抱",
    "227": "拍手",
    "232": "佛系",
    "240": "喷脸",
    "243": "甩头",
    "246": "加油抱抱",
    "262": "脑阔疼",
    "264": "捂脸",
    "265": "辣眼睛",
    "266": "哦哟",
    "267": "头秃",
    "268": "问号脸",
    "269": "暗中观察",
    "270": "emm",
    "271": "吃瓜",
    "272": "呵呵哒",
    "273": "我酸了",
    "277": "汪汪",
    "278": "汗",
    "281": "无眼笑",
    "282": "敬礼",
    "284": "面无表情",
    "285": "摸鱼",
    "287": "哦",
    "289": "睁眼",
    "290": "敲开心",
    "293": "摸锦鲤",
    "294": "期待",
    "297": "拜谢",
    "298": "元宝",
    "299": "牛啊",
    "305": "右亲亲",
    "306": "牛气冲天",
    "307": "喵喵",
    "314": "仔细分析",
    "315": "加油",
    "318": "崇拜",
    "319": "比心",
    "320": "庆祝",
    "322": "拒绝",
    "324": "吃糖",
    "326": "生气",
}


# Known special face IDs used by some OneBot/CQ implementations.
# These are not always delivered as `face` by every implementation.
SPECIAL_FACE_IDS: Dict[str, str] = {
    "358": "骰子",
    "359": "猜拳",
}


def _strip_brackets(s: str) -> str:
    s = s.strip()
    if len(s) >= 2:
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("【") and s.endswith("】")):
            s = s[1:-1].strip()
    return s


def _maybe_face_name(text: str) -> Optional[str]:
    """Heuristic: decide whether a string looks like a face name."""
    if not isinstance(text, str):
        return None
    t = _strip_brackets(text)
    if not t:
        return None
    # Too long is likely not a face name.
    if len(t) > 20:
        return None
    # Filter out strings that are purely digits/punctuation.
    if all(ch.isdigit() or ch in "-_:,." for ch in t):
        return None
    return t


def _extract_face_desc_from_raw(raw: Any) -> Optional[str]:
    """Try to extract a readable face name from `data.raw`.

    NapCatQQ documents `raw` as the original face payload (optional). In many
    deployments this contains the original face text.
    """
    if raw is None:
        return None

    # Direct string
    if isinstance(raw, str):
        return _maybe_face_name(raw)

    # Dict: try common fields first
    if isinstance(raw, dict):
        for k in ("text", "faceText", "face_text", "name", "desc", "summary"):
            if k in raw:
                v = raw.get(k)
                if isinstance(v, str):
                    cand = _maybe_face_name(v)
                    if cand:
                        return cand

        # Recursively search for the first plausible string
        for v in raw.values():
            cand = _extract_face_desc_from_raw(v)
            if cand:
                return cand

    # List/tuple: recurse
    if isinstance(raw, (list, tuple)):
        for v in raw:
            cand = _extract_face_desc_from_raw(v)
            if cand:
                return cand

    return None


def _resolve_face_desc(face_id: str, raw: Any = None) -> Optional[str]:
    """Resolve a face id into a human-readable description."""
    # 0) Specials (dice/rps)
    if face_id in SPECIAL_FACE_IDS:
        return SPECIAL_FACE_IDS[face_id]

    # 1) Raw payload (most reliable)
    raw_desc = _extract_face_desc_from_raw(raw)
    if raw_desc:
        return raw_desc

    # 2) User mapping (highest priority among tables)
    if face_id in QQ_FACE_MAPPING_USER:
        return QQ_FACE_MAPPING_USER.get(face_id)

    mode = QQ_FACE_MAP_MODE
    if mode not in {"official", "legacy", "auto"}:
        mode = "official"

    # 3) Mapping tables
    if mode == "legacy":
        return QQ_FACE_MAPPING_LEGACY.get(face_id) or QQ_FACE_MAPPING_OFFICIAL.get(face_id)
    if mode == "auto":
        # Try official first (latest desktop QQ), then legacy.
        return QQ_FACE_MAPPING_OFFICIAL.get(face_id) or QQ_FACE_MAPPING_LEGACY.get(face_id)
    # default: official
    return QQ_FACE_MAPPING_OFFICIAL.get(face_id) or QQ_FACE_MAPPING_LEGACY.get(face_id)


def _format_mface(data: dict) -> str:
    """Format a QQ marketplace emoticon (mface) into text."""
    summary = data.get("summary")
    emoji_id = data.get("emoji_id")
    pkg_id = data.get("emoji_package_id")

    label = None
    if isinstance(summary, str) and summary.strip():
        label = summary.strip()
    elif emoji_id:
        label = f"emoji_id={emoji_id}"
    else:
        label = "mface"

    # Keep package id if present (helps disambiguate)
    if pkg_id:
        return f" [商城表情:{label}; pkg={pkg_id}] "
    return f" [商城表情:{label}] "


def _format_rps_result(result: Any) -> str:
    """Rock-paper-scissors result mapping: 1=rock, 2=scissors, 3=paper."""
    try:
        r = int(str(result))
    except Exception:
        return str(result)

    return {1: "石头", 2: "剪刀", 3: "布"}.get(r, str(result))


def parse_onebot_array_msg(message_data: list | dict) -> Tuple[str, List[str], Optional[str]]:
    """Parse OneBot v11 message array.

    Returns:
        (plain_text, image_urls, reply_message_id)

    Notes:
        - QQ marketplace emoticons (mface) are often converted to `image` on
          receive. We detect those by `emoji_id` / `emoji_package_id` / `key`.
        - Unknown segment types are preserved as placeholders instead of being
          dropped.
    """

    text_content: str = ""
    image_urls: List[str] = []
    reply_id: Optional[str] = None

    # Compat: if a single dict is provided, wrap it
    if isinstance(message_data, dict):
        message_data = [message_data]

    if not isinstance(message_data, list):
        return "", [], None

    for segment in message_data:
        if not isinstance(segment, dict):
            continue

        msg_type = segment.get("type")
        data = segment.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        # ---------------------------
        # Text
        # ---------------------------
        if msg_type == "text":
            text_content += str(data.get("text", ""))
            continue

        # ---------------------------
        # Image (including mface->image)
        # ---------------------------
        if msg_type == "image":
            url = data.get("url")
            if isinstance(url, str) and url.strip():
                image_urls.append(url.strip())

            # If it looks like an mface converted into image, prefer sticker label.
            if any(k in data for k in ("emoji_id", "emoji_package_id", "key")):
                summary = data.get("summary")
                label = None
                if isinstance(summary, str) and summary.strip():
                    label = summary.strip()
                elif data.get("emoji_id"):
                    label = f"emoji_id={data.get('emoji_id')}"
                else:
                    label = "mface"
                text_content += f" [商城表情:{label}] "
            else:
                text_content += " [图片] "
            continue

        # ---------------------------
        # QQ system face
        # ---------------------------
        if msg_type == "face":
            face_id = str(data.get("id", "")).strip()

            # Some implementations may deliver dice/rps as face+resultId.
            if face_id in {"358", "359"} and data.get("resultId") is not None:
                if face_id == "358":
                    text_content += f" [骰子:{data.get('resultId')}] "
                else:
                    text_content += f" [猜拳:{_format_rps_result(data.get('resultId'))}] "
                continue

            face_desc = _resolve_face_desc(face_id, raw=data.get("raw"))
            if face_desc:
                text_content += f" [表情:{face_desc}] "
            else:
                text_content += f" [Face:{face_id}] "
            continue

        # ---------------------------
        # Marketplace face (NapCat: send type; receive often becomes image)
        # ---------------------------
        if msg_type == "mface":
            text_content += _format_mface(data)
            continue

        # ---------------------------
        # Dice / RPS / Poke
        # ---------------------------
        if msg_type == "dice":
            text_content += f" [骰子:{data.get('result', '')}] "
            continue

        if msg_type == "rps":
            text_content += f" [猜拳:{_format_rps_result(data.get('result'))}] "
            continue

        if msg_type == "poke":
            poke_type = data.get("type")
            poke_id = data.get("id")
            text_content += f" [戳一戳:type={poke_type}, id={poke_id}] "
            continue

        # ---------------------------
        # Mentions / reply
        # ---------------------------
        if msg_type == "at":
            qq = data.get("qq")
            text_content += f"[Mention:{qq}]"
            continue

        if msg_type == "reply":
            reply_id = str(data.get("id")) if data.get("id") is not None else None
            continue

        # ---------------------------
        # Other common message types
        # ---------------------------
        if msg_type == "record":
            text_content += " [语音消息] "
            continue

        if msg_type == "video":
            text_content += " [视频消息] "
            continue

        if msg_type == "file":
            name = data.get("name") or data.get("file") or "file"
            text_content += f" [文件:{name}] "
            continue

        if msg_type == "json":
            text_content += " [卡片消息/小程序] "
            continue

        if msg_type == "xml":
            text_content += " [XML消息] "
            continue

        # Fallback: preserve unknown segment types
        if msg_type:
            text_content += f" [{msg_type}] "

    return text_content.strip(), image_urls, reply_id
