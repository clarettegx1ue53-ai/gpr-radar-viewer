from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE_DIR = Path(r"F:/导出图像")
OUTPUT_DIR = SOURCE_DIR / "补全坐标"
LEFT_PAD = 170

PLOT_X0 = 110
PLOT_Y0 = 72
CHANNEL_HEIGHT = 256
CHANNEL_COUNT = 10
PLOT_Y1 = PLOT_Y0 + CHANNEL_HEIGHT * CHANNEL_COUNT
X_AXIS_Y = 2644

START_STATIONS = {
    "06": 234_400,
    "07": 233_400,
    "08": 232_400,
    "09": 231_400,
}

BG = (245, 245, 245)
INK = (55, 55, 55)


def station_label(metres: int) -> str:
    kilometres, remainder = divmod(metres, 1000)
    return f"K{kilometres}+{remainder:03d}"


def rotated_label(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    box = font.getbbox(text)
    layer = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((4 - box[0], 4 - box[1]), text, font=font, fill=INK + (255,))
    return layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)


def add_axes(source: Path, output: Path, start_station: int) -> None:
    with Image.open(source) as opened:
        original = opened.convert("RGB")

    canvas = Image.new("RGB", (original.width + LEFT_PAD, original.height), BG)
    canvas.paste(original, (LEFT_PAD, 0))
    draw = ImageDraw.Draw(canvas)

    plot_left = LEFT_PAD + PLOT_X0
    plot_right = LEFT_PAD + original.width - 17

    # Replace the old channel-name gutter, then draw a full depth axis for each channel.
    draw.rectangle((LEFT_PAD, PLOT_Y0, plot_left - 1, PLOT_Y1 - 1), fill=BG)

    # A compact font keeps the 1.20 m tick, 1.25 m endpoint, and the next
    # channel's 0.00 m origin independently legible in the 256 px channel.
    label_font = ImageFont.truetype(r"C:/Windows/Fonts/msyh.ttc", 10)
    channel_font = ImageFont.truetype(r"C:/Windows/Fonts/msyh.ttc", 16)
    unit_font = ImageFont.truetype(r"C:/Windows/Fonts/msyh.ttc", 15)
    x_font = ImageFont.truetype(r"C:/Windows/Fonts/msyh.ttc", 15)
    unit_image = rotated_label("深度 (m)", unit_font)

    depths = [round(i / 10, 1) for i in range(13)] + [1.25]
    for channel in range(CHANNEL_COUNT):
        top = PLOT_Y0 + channel * CHANNEL_HEIGHT
        bottom = top + CHANNEL_HEIGHT

        draw.text((8, (top + bottom) / 2), f"CH{channel + 1:03d}", font=channel_font, fill=INK, anchor="lm")
        unit_y = int((top + bottom - unit_image.height) / 2)
        canvas.paste(unit_image, (105, unit_y), unit_image)

        draw.line((plot_left - 1, top, plot_left - 1, bottom), fill=INK, width=1)
        for depth in depths:
            y = top + depth / 1.25 * CHANNEL_HEIGHT
            draw.line((plot_left - 9, round(y), plot_left - 1, round(y)), fill=INK, width=1)
            text = "1.25" if depth == 1.25 else f"{depth:.1f}"
            # At an internal boundary, keep the previous channel's 1.25 label
            # on the boundary and move the next channel's 0.0 label slightly
            # inward. This also leaves room for the nearby 1.2/0.1 labels.
            if depth == 0 and channel > 0:
                text_y = y + 9
            else:
                text_y = y
            draw.text((plot_left - 14, text_y), text, font=label_font, fill=INK, anchor="rm")

    # Replace the original 50 m axis with a labelled 10 m axis.
    draw.rectangle((plot_left, X_AXIS_Y, plot_right, 2683), fill=BG)
    draw.line((plot_left, X_AXIS_Y, plot_right, X_AXIS_Y), fill=INK, width=2)
    span = plot_right - plot_left
    for offset in range(0, 1001, 10):
        x = plot_left + span * offset / 1000
        draw.line((round(x), X_AXIS_Y, round(x), X_AXIS_Y + 9), fill=INK, width=1)
        label = station_label(start_station - offset)
        anchor = "la" if offset == 0 else "ra" if offset == 1000 else "ma"
        draw.text((x, X_AXIS_Y + 14), label, font=x_font, fill=INK, anchor=anchor)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    sources = sorted(SOURCE_DIR.glob("2026-09-004-??_降噪增强_已标注.png"))
    if len(sources) != 4:
        raise RuntimeError(f"应找到4张源图，实际找到{len(sources)}张")

    for source in sources:
        image_id = source.name.split("_")[0].rsplit("-", 1)[-1]
        if image_id not in START_STATIONS:
            raise RuntimeError(f"未配置起始桩号：{source.name}")
        output = OUTPUT_DIR / f"{source.stem}_补全坐标.png"
        add_axes(source, output, START_STATIONS[image_id])
        print(output)


if __name__ == "__main__":
    main()
