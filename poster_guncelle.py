# -*- coding: utf-8 -*-
"""Güncel tez/poster PDF'ini üretir.

Orijinal poster.pdf dosyasına dokunmaz; güncel sürümü Downloads klasörüne
poster_guncel.pdf olarak yazar.
"""

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "grafikler"
OUT_PDF = Path(r"C:\Users\gulsum\Downloads\poster_guncel.pdf")
INTERFACE_IMG = GRAPH_DIR / "poster_dss_interface_snapshot.png"
DATA_IMG = GRAPH_DIR / "poster_data_management_snapshot.png"

BLUE_DARK = "#1e3a8a"
BLUE = "#2563eb"
BLUE_MID = "#3b82f6"
BLUE_LIGHT = "#dbeafe"
INK = "#0f172a"
MUTED = "#475569"
RED = "#ff4757"
GREEN = "#22c55e"
BG_DARK = "#0b0f17"
PANEL_DARK = "#10151f"
INPUT_DARK = "#282832"

PDF_FONT = "ArialTR"
PDF_FONT_BOLD = "ArialTR-Bold"


def font(size=24, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_round_rect(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def make_interface_snapshot(path: Path):
    img = Image.new("RGB", (1500, 760), BG_DARK)
    d = ImageDraw.Draw(img)
    f_title = font(36, True)
    f_label = font(19, True)
    f_text = font(24)
    f_metric = font(43)
    f_table = font(18)

    d.text((35, 35), "📋 Arıza Giriş Formu", fill="white", font=f_title)
    d.text((650, 35), "🎯 Tahmin Sonuçları", fill="white", font=f_title)

    d.text((35, 105), "Makine Ünitesi", fill="white", font=f_label)
    draw_round_rect(d, (35, 135, 570, 183), 8, INPUT_DARK)
    d.text((55, 146), "Boru Lazer", fill="white", font=f_text)

    d.text((35, 225), "Arıza Açıklaması", fill="white", font=f_label)
    draw_round_rect(d, (35, 255, 570, 400), 8, INPUT_DARK)
    d.text((55, 278), "ısınma var", fill="white", font=f_text)
    draw_round_rect(d, (35, 425, 570, 475), 8, RED)
    d.text((225, 438), "ANALİZ ET VE TAHMİN ET", fill="white", font=f_label)

    metrics = [
        ("Beklenen Süre", "35.9 dk"),
        ("Planlama Süresi", "45.9 dk"),
        ("Uzun Duruş Riski", "%9"),
        ("Güven", "Yüksek"),
    ]
    x0 = 650
    for i, (label, value) in enumerate(metrics):
        x = x0 + i * 210
        d.text((x, 105), label, fill="white", font=f_label)
        d.text((x, 135), value, fill="white", font=f_metric)

    draw_round_rect(d, (650, 215, 1455, 285), 8, "#123f2a")
    d.ellipse((675, 238, 697, 260), fill=GREEN)
    d.text((715, 238), "NORMAL: Kısa süreli müdahale olasılığı daha yüksek.", fill="#4ade80", font=f_text)

    d.text((650, 330), "Benzer Geçmiş Özeti", fill="white", font=f_title)
    summary = [("Benzer Kayıt", "6"), ("Medyan", "31.0 dk"), ("P80", "42.0 dk"), ("Maksimum", "42.0 dk")]
    for i, (label, value) in enumerate(summary):
        x = 650 + i * 210
        d.text((x, 390), label, fill="white", font=f_label)
        d.text((x, 420), value, fill="white", font=f_metric)

    d.text((650, 525), "Benzer Geçmiş Kayıtlar", fill="white", font=f_title)
    x_cols = [650, 790, 970, 1285]
    headers = ["Benzerlik", "Makine", "Arıza", "Süre (dk)"]
    for x, h in zip(x_cols, headers):
        d.text((x, 585), h, fill="#cbd5e1", font=f_table)
    rows = [
        ("0.409", "Sac Lazer 15kW", "Isınmadan kaynaklı arıza 28 dk", "28"),
        ("0.409", "Sac Lazer 15kW", "Isınmadan kaynaklı arıza 15 dk", "22"),
        ("0.409", "Sac Lazer 15kW", "Isınmadan kaynaklı arıza", "32"),
    ]
    y = 625
    for row in rows:
        for x, value in zip(x_cols, row):
            d.text((x, y), value, fill="white", font=f_table)
        y += 42

    path.parent.mkdir(exist_ok=True)
    img.save(path)


def make_data_snapshot(path: Path):
    img = Image.new("RGB", (1500, 520), BG_DARK)
    d = ImageDraw.Draw(img)
    f_title = font(32, True)
    f_label = font(19, True)
    f_text = font(23)

    d.text((35, 28), "🗂️ Veri Yönetimi (Yeni Kayıt / Kayıt Düzenleme)", fill="white", font=f_title)
    d.text((35, 90), "Veri dosyası: verileriniz_normalize.xlsx | Toplam kayıt: 1500", fill="#cbd5e1", font=f_text)
    draw_round_rect(d, (35, 135, 1465, 200), 8, "#143250")
    d.text((55, 155), "Bu bölüm Excel verisini günceller. Yeni kayıtların model tahminlerine yansıması için modelin yeniden eğitilmesi gerekir.", fill="#60a5fa", font=f_text)

    d.text((35, 240), "Yeni Kayıt Ekle", fill=RED, font=f_label)
    d.text((170, 240), "Kayıt Düzenle", fill="white", font=f_label)
    d.line((35, 270, 1465, 270), fill="#2b313c", width=2)
    d.line((35, 270, 145, 270), fill=RED, width=4)

    fields = [
        ("Tarih", "2026/05/16"),
        ("Makine Ünitesi", "Boru Lazer"),
        ("Arıza Açıklaması", ""),
        ("Süre (dk)", "30"),
    ]
    y = 300
    for label, value in fields:
        d.text((55, y), label, fill="white", font=f_label)
        h = 48 if label != "Arıza Açıklaması" else 95
        draw_round_rect(d, (55, y + 28, 1445, y + 28 + h), 8, INPUT_DARK)
        if value:
            d.text((75, y + 40), value, fill="white", font=f_text)
        y += h + 65
        if y > 500:
            break

    img.save(path)


def add_wrapped(c, text, x, y, width, font_name="Helvetica", font_size=8.5, leading=10.5, color=INK):
    c.setFillColor(colors.HexColor(color))
    c.setFont(font_name, font_size)
    max_chars = max(20, int(width / (font_size * 0.48)))
    for line in text.split("\n"):
        wrapped = wrap(line, max_chars) or [""]
        for item in wrapped:
            c.drawString(x, y, item)
            y -= leading
    return y


def register_pdf_fonts():
    regular = Path(r"C:\Windows\Fonts\arial.ttf")
    bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont(PDF_FONT, str(regular)))
        pdfmetrics.registerFont(TTFont(PDF_FONT_BOLD, str(bold)))
        return PDF_FONT, PDF_FONT_BOLD
    return "Helvetica", "Helvetica-Bold"


def draw_section(c, title, x, y, w, h):
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    c.setFillColor(colors.HexColor(BLUE_DARK))
    c.setFont(PDF_FONT_BOLD, 11)
    c.drawString(x + 10, y + h - 17, title)
    c.setStrokeColor(colors.HexColor(BLUE_LIGHT))
    c.setLineWidth(1)
    c.line(x + 10, y + h - 24, x + w - 10, y + h - 24)


def draw_image(c, img_path, x, y, w, h):
    c.drawImage(ImageReader(str(img_path)), x, y, width=w, height=h, preserveAspectRatio=True, anchor="c")


def main():
    regular_font, bold_font = register_pdf_fonts()
    make_interface_snapshot(INTERFACE_IMG)
    make_data_snapshot(DATA_IMG)

    c = canvas.Canvas(str(OUT_PDF), pagesize=A4)
    page_w, page_h = A4

    c.setFillColor(colors.white)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    c.setFillColor(colors.HexColor(BLUE_DARK))
    c.rect(0, page_h - 78, page_w, 78, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(bold_font, 18)
    c.drawCentredString(page_w / 2, page_h - 28, "MACHINE DOWNTIME DURATION PREDICTION")
    c.setFont(bold_font, 13)
    c.drawCentredString(page_w / 2, page_h - 48, "NLP and Gradient Boosting Based Decision Support System")
    c.setFont(regular_font, 8)
    c.drawCentredString(page_w / 2, page_h - 65, "Berkay Koçak, Gamze Sevimli, Ümmügülsüm Çiftçi | Advisor: Asst. Prof. Sena Aydoğan")

    margin = 24
    gap = 12
    col_w = (page_w - 2 * margin - gap) / 2
    top = page_h - 95

    draw_section(c, "Aim and Model Workflow", margin, top - 126, col_w, 126)
    y = top - 30
    y = add_wrapped(
        c,
        "This study develops a decision support system that predicts machine downtime duration from operator maintenance logs. "
        "The final prediction model uses TF-IDF text features, machine type information and a Gradient Boosting Regressor.",
        margin + 10,
        y,
        col_w - 20,
        font_name=regular_font,
    )
    y -= 5
    add_wrapped(
        c,
        "Input: machine unit + failure description\nOutput: expected duration, planning duration, long downtime risk, confidence level and similar historical records.",
        margin + 10,
        y,
        col_w - 20,
        font_name=regular_font,
    )

    draw_section(c, "Final Model Performance", margin + col_w + gap, top - 126, col_w, 126)
    x = margin + col_w + gap + 12
    y0 = top - 48
    metrics = [("MAE", "11.64 min"), ("Median AE", "8.52 min"), ("Risk AUC", "0.976"), ("Risk Accuracy", "93.0%")]
    for i, (k, v) in enumerate(metrics):
        mx = x + (i % 2) * 135
        my = y0 - (i // 2) * 46
        c.setFillColor(colors.HexColor(BLUE))
        c.setFont(bold_font, 9)
        c.drawString(mx, my, k)
        c.setFillColor(colors.HexColor(INK))
        c.setFont(bold_font, 18)
        c.drawString(mx, my - 20, v)

    draw_section(c, "DSS Interface Output", margin, top - 330, page_w - 2 * margin, 188)
    draw_image(c, INTERFACE_IMG, margin + 10, top - 320, page_w - 2 * margin - 20, 145)
    add_wrapped(
        c,
        "The interface converts the entered failure note into a downtime prediction and supports planning with similar historical records.",
        margin + 12,
        top - 172,
        page_w - 2 * margin - 24,
        font_size=8,
        font_name=regular_font,
    )

    draw_section(c, "Model Diagnostics", margin, top - 560, col_w, 214)
    draw_image(c, GRAPH_DIR / "00_model_performance_overview.png", margin + 8, top - 540, col_w - 16, 142)
    add_wrapped(
        c,
        "Actual-predicted alignment and absolute error distribution show the regression behavior of the final Gradient Boosting model.",
        margin + 10,
        top - 390,
        col_w - 20,
        font_size=8,
        font_name=regular_font,
    )

    draw_section(c, "Long Downtime Risk Model", margin + col_w + gap, top - 560, col_w, 214)
    draw_image(c, GRAPH_DIR / "07_long_downtime_roc_curve.png", margin + col_w + gap + 14, top - 532, col_w - 28, 132)
    add_wrapped(
        c,
        "The ROC curve indicates high discrimination for identifying failures with long downtime risk (AUC = 0.976).",
        margin + col_w + gap + 10,
        top - 390,
        col_w - 20,
        font_size=8,
        font_name=regular_font,
    )

    draw_section(c, "Data Update Module", margin, 38, col_w, 164)
    draw_image(c, DATA_IMG, margin + 8, 72, col_w - 16, 104)
    add_wrapped(
        c,
        "New and edited records are written to the Excel dataset. The model can then be retrained to reflect updated historical data.",
        margin + 10,
        62,
        col_w - 20,
        font_size=7.8,
        font_name=regular_font,
    )

    draw_section(c, "Conclusion", margin + col_w + gap, 38, col_w, 164)
    add_wrapped(
        c,
        "Among tested alternatives, Gradient Boosting was selected as the final DSS model because it achieved the lowest MAE among the selected machine learning models. "
        "The system combines duration prediction, risk estimation and similar-case retrieval to support production planning decisions.",
        margin + col_w + gap + 10,
        172,
        col_w - 20,
        font_name=regular_font,
    )
    c.setFillColor(colors.HexColor(BLUE_DARK))
    c.setFont(bold_font, 8)
    c.drawString(margin + col_w + gap + 10, 55, "Final DSS model: Gradient Boosting Regressor")

    c.save()
    print(f"Updated poster written to: {OUT_PDF}")


if __name__ == "__main__":
    main()
