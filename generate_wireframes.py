from PIL import Image, ImageDraw, ImageFont
import os

font_path = "arial.ttf"

def get_font(size):
    try:
        return ImageFont.truetype(font_path, size)
    except IOError:
        return ImageFont.load_default()

def draw_base_layout(draw, width, height, title):
    font_large = get_font(24)
    font_med = get_font(18)
    
    # Background (grayish)
    draw.rectangle([0, 0, width, height], fill="#f0f0f0")
    
    # Title
    draw.text((20, 20), title, fill="#333333", font=font_large)

    # Left: Image
    img_box = [20, 60, width*0.5, height-40]
    draw.rectangle(img_box, outline="#666666", width=2, fill="#e0e0e0")
    draw.text((img_box[0]+20, img_box[1]+20), "Image Region\n(Original / Processed)", fill="#333", font=font_med)

    # Middle: Controls
    ctrl_box = [width*0.5 + 20, 60, width*0.75, height-40]
    draw.rectangle(ctrl_box, outline="#666666", width=2, fill="#e6e6e6")
    draw.text((ctrl_box[0]+20, ctrl_box[1]+20), "Control Panel", fill="#333", font=font_med)

    # Right: Results
    res_box = [width*0.75 + 20, 60, width-20, height-40]
    draw.rectangle(res_box, outline="#666666", width=2, fill="#e6e6e6")
    draw.text((res_box[0]+20, res_box[1]+20), "Results / Tables", fill="#333", font=font_med)

    # Bottom Stage: Status Bar
    status_box = [0, height-30, width, height]
    draw.rectangle(status_box, outline="#666666", width=1, fill="#d0d0d0")
    draw.text((10, height-25), "[Status] Ready...", fill="#333", font=get_font(14))

    return img_box, ctrl_box, res_box

def draw_control(draw, x, y, label, type="slider", annotated=None):
    font = get_font(14)
    font_anno = get_font(14)
    draw.text((x, y), label, fill="#000", font=font)
    if type == "slider":
        draw.line((x, y+20, x+120, y+20), fill="#666", width=3)
        draw.ellipse((x+50, y+15, x+60, y+25), fill="#999")
    elif type == "button":
        draw.rectangle([x, y+15, x+80, y+35], fill="#ccc", outline="#333")
        draw.text((x+10, y+18), "CLICK", fill="#000", font=font)
    elif type == "dropdown":
        draw.rectangle([x, y+15, x+120, y+35], fill="#fff", outline="#333")
        draw.text((x+5, y+18), "v Option", fill="#000", font=font)
    elif type == "checkbox":
        draw.rectangle([x, y+15, x+15, y+30], fill="#fff", outline="#333")
        draw.text((x+25, y+15), "Enable", fill="#000", font=font)

    if annotated:
        draw.text((x + 130, y+15), f"ID: {annotated}", fill="#c00000", font=font_anno)


def generate_minimalist():
    w, h = 1280, 720
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    _, ctrl_box, _ = draw_base_layout(draw, w, h, "1. Minimalist Wireframe")

    cx = ctrl_box[0] + 20
    cy = ctrl_box[1] + 60

    draw_control(draw, cx, cy, "Algorithm Selection", "dropdown")
    draw_control(draw, cx, cy+60, "Sensitivity", "slider")
    draw_control(draw, cx, cy+120, "Run Detection", "button")

    img.save("wireframe_minimalist.png")

def generate_parameter_rich():
    w, h = 1280, 720
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    _, ctrl_box, _ = draw_base_layout(draw, w, h, "2. Parameter-Rich Wireframe")

    cx = ctrl_box[0] + 20
    cy = ctrl_box[1] + 60

    draw_control(draw, cx, cy, "Algorithm Selection", "dropdown")
    draw_control(draw, cx, cy+50, "Canny Low Thresh", "slider")
    draw_control(draw, cx, cy+100, "Canny High Thresh", "slider")
    draw_control(draw, cx, cy+150, "Hough Param 1", "slider")
    draw_control(draw, cx, cy+200, "Hough Param 2", "slider")
    draw_control(draw, cx, cy+250, "Min Radius", "slider")
    draw_control(draw, cx, cy+300, "Max Radius", "slider")
    draw_control(draw, cx, cy+350, "Run Detection", "button")

    img.save("wireframe_parameter_rich.png")

def generate_analyst():
    w, h = 1280, 720
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    _, ctrl_box, res_box = draw_base_layout(draw, w, h, "3. Analyst-Oriented Wireframe")

    cx = ctrl_box[0] + 20
    cy = ctrl_box[1] + 60

    draw_control(draw, cx, cy, "Compare Algorithms", "checkbox")
    draw_control(draw, cx, cy+50, "Auto-tune weights", "checkbox")
    draw_control(draw, cx, cy+100, "Run Batch Profile", "button")

    # Draw a mock chart in the Results
    rx = res_box[0] + 20
    ry = res_box[1] + 60
    draw.rectangle([rx, ry, rx+200, ry+150], fill="#fff", outline="#333")
    draw.text((rx+10, ry+10), "IOU / Metrics Chart", fill="#333", font=get_font(14))
    draw.line((rx+20, ry+140, rx+180, ry+20), fill="#00f", width=2)
    
    # Table data
    draw.text((rx, ry+170), "Detected Circles: 42\nFalse Positives: 2\nAvg Time: 1.2s", fill="#000", font=get_font(14))

    img.save("wireframe_analyst.png")


def generate_final_annotated():
    w, h = 1280, 720
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    _, ctrl_box, _ = draw_base_layout(draw, w, h, "Final Annotated Wireframe (Parameter-Rich Hybrid)")

    cx = ctrl_box[0] + 20
    cy = ctrl_box[1] + 60

    draw_control(draw, cx, cy, "Algorithm Selection", "dropdown", "dropdown_algo")
    draw_control(draw, cx, cy+50, "Canny High Thresh", "slider", "sld_canny_high")
    draw_control(draw, cx, cy+100, "Max Radius", "slider", "sld_max_radius")
    draw_control(draw, cx, cy+150, "Run Detection", "button", "btn_run")
    draw_control(draw, cx, cy+200, "Compare Mode", "checkbox", "chk_compare")
    draw_control(draw, cx, cy+250, "Export Results", "button", "btn_export")

    img.save("wireframe_final_annotated.png")

if __name__ == "__main__":
    generate_minimalist()
    generate_parameter_rich()
    generate_analyst()
    generate_final_annotated()
    print("All wireframes generated.")
