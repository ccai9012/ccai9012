"""Generate compact teaching diagrams used by the WIP course notebooks."""

from html import escape
from pathlib import Path


OUT = Path(__file__).parent
NAVY = "#0b1f33"
BLUE = "#2374a5"
CYAN = "#00b5e2"
PINK = "#ef1c88"
PALE = "#eaf5fa"
LIGHT = "#fff9fc"
GRAY = "#526777"


def text(x, y, value, size=19, color=NAVY, weight="normal", anchor="start", font="Georgia,serif"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'fill="{color}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}">{escape(value)}</text>'
    )


def rect(x, y, w, h, fill="#fff", stroke=BLUE, radius=16, width=2):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def line(x1, y1, x2, y2, color=BLUE, width=3, arrow=True):
    mark = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round"{mark}/>'
    )


def card(x, y, w, h, heading, detail, fill=PALE):
    return "".join(
        [
            rect(x, y, w, h, fill=fill),
            text(x + 18, y + 37, heading, 20, weight="bold"),
            text(x + 18, y + 68, detail, 16, color=GRAY),
        ]
    )


def save(name, title, subtitle, drawing, height=430):
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" '
        f'role="img" aria-label="{escape(title)}. {escape(subtitle)}">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0 0 L10 5 L0 10z" fill="{BLUE}"/></marker></defs>'
        f'<rect width="1200" height="{height}" fill="#fff"/>'
        + text(42, 47, title, 27, weight="bold")
        + text(42, 77, subtitle, 16, color=GRAY)
        + drawing
        + "</svg>"
    )
    (OUT / name).write_text(content, encoding="utf-8")


# Use the robot-story prompt from the LLM Basics notebook. Probabilities are
# illustrative, not values returned by the DeepSeek API.
bars = text(55, 126, 'Same story context: "The robot picked up a ..."', 21, weight="bold")
for offset, heading, vals, tint in [
    (55, "Low temperature: one choice dominates", [0.74, 0.15, 0.08, 0.03], BLUE),
    (642, "High temperature: alternatives get more weight", [0.38, 0.29, 0.21, 0.12], PINK),
]:
    bars += rect(offset, 151, 503, 225, fill="#fff", stroke="#a8c2d1", radius=9)
    bars += text(offset + 20, 181, heading, 18, weight="bold")
    for j, (label, val) in enumerate(zip(["brush", "pencil", "canvas", "spoon"], vals)):
        y = 208 + j * 39
        bars += text(offset + 24, y + 17, label, 17)
        bars += rect(offset + 104, y, int(val * 360), 22, fill=tint, stroke="none", radius=3, width=0)
        bars += text(offset + 465, y + 17, f"{val:.0%}", 15, color=GRAY, anchor="end")
save(
    "temperature_distributions.svg",
    "One robot story, different sampling temperatures",
    "Illustrative next-token probabilities, not measured DeepSeek output; temperature does not check truth.",
    bars,
)

cnn = text(55, 128, "The notebook hooks two convolution layers for one held-out Fashion-MNIST image", 20, weight="bold")
cnn += rect(72, 160, 174, 160, fill="#e3e3e3", stroke="#a8c2d1", radius=5)
cnn += '<path d="M99 256 Q115 233 132 239 L156 257 L200 264 L219 288 Q161 304 107 288Z" fill="#526777"/>'
cnn += text(159, 348, "input: 28 × 28", 17, anchor="middle")
cnn += line(252, 241, 344, 241)
for group_x, heading, detail, size, tint in [
    (350, "conv1", "32 maps · 28 × 28", 22, CYAN),
    (752, "conv2", "64 maps · 14 × 14", 17, PINK),
]:
    cnn += text(group_x, 173, heading, 21, weight="bold")
    cnn += text(group_x, 201, detail, 16, color=GRAY)
    for k in range(3):
        ox, oy = group_x + k * 63, 220 + (k % 2) * 13
        cnn += rect(ox, oy, 4 * size + 8, 4 * size + 8, fill="#fff", stroke=tint, radius=3)
        for row in range(4):
            for col in range(4):
                opacity = [0.18, 0.38, 0.63, 0.88][(row * 2 + col + k) % 4]
                cnn += f'<rect x="{ox+5+col*size}" y="{oy+5+row*size}" width="{size-2}" height="{size-2}" fill="{tint}" opacity="{opacity}"/>'
cnn += line(664, 241, 739, 241)
cnn += text(55, 392, "Compare individual channels first; mean and max maps are summaries of those channels.", 17, color=GRAY)
save("cnn_feature_maps.svg", "One image, many learned response maps", "The input drawing is schematic; the notebook displays real held-out pixels and captured conv1/conv2 activations.", cnn)

embed = text(52, 122, "1  An embedding gives each word a learned list of numbers", 21, weight="bold")
embed += rect(53, 145, 508, 142, fill=PALE, stroke="#a8c2d1", radius=10)
embed += text(77, 184, "doctor", 22, weight="bold")
embed += text(175, 184, "→  [ 0.2, −0.4, … ]", 21)
embed += text(77, 233, "nurse", 22, weight="bold")
embed += text(175, 233, "→  [ 0.3, −0.3, … ]", 21)
embed += text(78, 266, "Numbers are illustrative; real vectors have many dimensions.", 15, color=GRAY)
embed += line(569, 215, 640, 215)
embed += text(672, 164, "Similar use can yield nearby vectors", 19, weight="bold")
for x, y, label, color in [
    (730, 217, "doctor", BLUE),
    (847, 238, "nurse", BLUE),
    (986, 211, "apple", PINK),
    (1088, 234, "banana", PINK),
]:
    embed += f'<circle cx="{x}" cy="{y}" r="8" fill="{color}"/>'
    embed += text(x, y - 16, label, 16, anchor="middle")
embed += text(682, 277, "Position is schematic, not a measured PCA plot.", 15, color=GRAY)

embed += '<path d="M52 307H1148" stroke="#d9e4eb" stroke-width="2"/>'
embed += text(52, 343, "2  Biased patterns in text can become associations in the vectors", 21, weight="bold")
embed += text(60, 379, "Repeated text patterns (toy)", 17, color=GRAY)
embed += text(62, 410, '"he … engineer"', 20, color=BLUE)
embed += text(62, 440, '"she … nurse"', 20, color=PINK)
embed += line(290, 412, 387, 412)
embed += text(410, 379, "Learned association", 17, color=GRAY)
embed += line(420, 420, 756, 420, color="#a8c2d1", width=2, arrow=False)
for x, label, color in [
    (458, "she", PINK),
    (523, "nurse", PINK),
    (665, "he", BLUE),
    (721, "engineer", BLUE),
]:
    embed += f'<circle cx="{x}" cy="420" r="8" fill="{color}"/>'
    embed += text(x, 451, label, 16, color=color, anchor="middle")
embed += line(763, 412, 823, 412)
embed += text(850, 381, "Possible bias", 19, weight="bold")
embed += text(850, 414, "A similarity score may echo", 17)
embed += text(850, 441, "a stereotype in the text.", 17)
embed += text(52, 486, "The notebook tests this model's associations; a word score does not describe anyone's ability or suitability.", 16, color=GRAY)
save(
    "embedding_bias_origin.svg",
    "What is a word embedding, and where can bias enter?",
    "Training text shapes word vectors; those vectors can preserve unwanted associations.",
    embed,
    height=510,
)

def lane(x, y, w, h):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="23" '
        f'fill="{LIGHT}" stroke="{PINK}" stroke-width="2.5" stroke-dasharray="9 7"/>'
    )


token = text(55, 126, 'Input string:  "Hello, world!"', 23, weight="bold")
token += text(55, 180, "GPT-2 token pieces", 18, color=GRAY)
token += text(55, 279, "Vocabulary IDs", 18, color=GRAY)
for x, w, piece, token_id, tint in [
    (275, 177, "Hello", "15496", BLUE),
    (471, 98, ",", "11", CYAN),
    (588, 233, "Ġworld", "995", PINK),
    (840, 98, "!", "0", BLUE),
]:
    token += rect(x, 145, w, 68, fill=PALE if tint != PINK else LIGHT, stroke=tint, radius=9)
    token += text(x + w / 2, 188, piece, 23, weight="bold", anchor="middle")
    token += line(x + w / 2, 216, x + w / 2, 264, color=tint, width=2)
    token += rect(x, 267, w, 54, fill="#fff", stroke=tint, radius=9)
    token += text(x + w / 2, 302, token_id, 21, anchor="middle", font="Menlo,monospace")
token += rect(280, 346, 660, 42, fill=LIGHT, stroke=PINK, radius=9)
token += text(610, 373, "Model input: [15496, 11, 995, 0]", 19, weight="bold", anchor="middle", font="Menlo,monospace")
save("tokenisation_pipeline.svg", "One sentence, four GPT-2 tokens", "Ġ in Ġworld displays a leading space; the IDs shown match the notebook's saved GPT-2 example.", token)

bpe = text(55, 126, 'The notebook example: "unbelievable"', 22, weight="bold")
bpe += text(55, 179, "As one whole word", 17, color=GRAY)
bpe += rect(285, 149, 570, 47, fill=PALE, stroke=BLUE, radius=6)
bpe += text(570, 180, "unbelievable", 21, anchor="middle")
bpe += text(894, 180, "1 piece if known", 17, color=GRAY)
bpe += text(55, 250, "As characters", 17, color=GRAY)
for j, ch in enumerate("unbelievable"):
    x = 285 + 48 * j
    bpe += rect(x, 220, 43, 48, fill="#fff", stroke="#a8c2d1", radius=5)
    bpe += text(x + 21, 251, ch, 19, anchor="middle")
bpe += text(894, 251, "12 tiny pieces", 17, color=GRAY)
bpe += text(55, 323, "GPT-2 output", 17, color=GRAY)
for x, w, piece in [(285, 105, "un"), (395, 150, "bel"), (550, 150, "iev"), (705, 150, "able")]:
    bpe += rect(x, 292, w, 48, fill=LIGHT, stroke=PINK, radius=5)
    bpe += text(x + w / 2, 323, piece, 19, weight="bold", anchor="middle")
bpe += text(894, 323, "4 reusable pieces", 17, color=GRAY)
save("subword_tradeoff.svg", "Compare three ways to split one word", "The GPT-2 row uses the actual pieces shown by this notebook; other tokenizers may split differently.", bpe)

ml = text(55, 122, "The MLP in this notebook maps eight housing features to one predicted value", 20, weight="bold")
for i in range(8):
    for j in range(6):
        ml += f'<path d="M144 {191+i*19} L436 {191+j*25}" stroke="{BLUE}" stroke-width="1" opacity="0.13"/>'
for j in range(6):
    ml += f'<path d="M454 {191+j*25} L747 191" stroke="{CYAN}" stroke-width="1.4" opacity="0.35"/>'
for x, count, tint in [(135, 8, BLUE), (445, 6, CYAN), (755, 1, PINK)]:
    for j in range(count):
        y = 191 + j * (19 if count == 8 else 25)
        ml += f'<circle cx="{x}" cy="{y}" r="8" fill="{tint}" opacity="0.85"/>'
ml += text(82, 371, "8 inputs", 17, weight="bold")
ml += text(386, 371, "64 hidden ReLU units", 17, weight="bold")
ml += text(707, 371, "predicted price", 17, weight="bold")
ml += rect(850, 160, 290, 183, fill="#fff", stroke="#a8c2d1", radius=8)
ml += text(870, 192, "Read the loss curves", 19, weight="bold")
ml += line(889, 310, 1109, 310, color=GRAY, width=1, arrow=False)
ml += line(889, 210, 889, 310, color=GRAY, width=1, arrow=False)
ml += '<path d="M896 221 Q945 265 982 281 Q1045 296 1102 300" fill="none" stroke="#2374a5" stroke-width="3"/>'
ml += '<path d="M896 227 Q945 267 982 277 Q1045 281 1102 281" fill="none" stroke="#ef1c88" stroke-width="3"/>'
ml += text(861, 367, "Blue: train    Pink: validation", 16, color=GRAY)
save("ml_housing_learning_map.svg", "Inside the housing-price MLP", "The 64-unit layer is compressed to six dots for clarity; loss curves are schematic, not saved training results.", ml)

ml101 = text(55, 119, "The same split supports two housing-price predictors", 20, weight="bold")
ml101 += rect(55, 164, 236, 191, fill=PALE, stroke=BLUE, radius=8)
ml101 += text(75, 199, "Housing CSV", 20, weight="bold")
ml101 += text(75, 236, "X: 8 features", 17)
ml101 += text(75, 266, "y: median value", 17)
ml101 += text(75, 319, "80% train / 20% test", 17, color=GRAY)
ml101 += line(296, 236, 350, 215)
ml101 += line(296, 281, 350, 314)
for y, heading, path_d, tint, detail in [
    (157, "Linear regression", "M566 226 L700 178", BLUE, "train-only scaling"),
    (270, "Decision tree", "M565 339 H605 V321 H652 V292 H700", PINK, "raw training features"),
]:
    ml101 += rect(357, y, 384, 95, fill="#fff", stroke="#a8c2d1", radius=8)
    ml101 += text(376, y + 34, heading, 19, weight="bold")
    ml101 += text(376, y + 65, detail, 16, color=GRAY)
    ml101 += f'<path d="{path_d}" fill="none" stroke="{tint}" stroke-width="4"/>'
ml101 += line(746, 211, 829, 238)
ml101 += line(746, 317, 829, 277)
ml101 += rect(835, 164, 310, 191, fill=LIGHT, stroke=PINK, radius=8)
ml101 += text(855, 199, "Held-out test rows", 19, weight="bold")
ml101 += text(855, 241, "Compare MSE and R²", 18)
ml101 += text(855, 281, "Plot true vs predicted", 17)
ml101 += text(855, 319, "Inspect individual errors", 16, color=GRAY)
ml101 += text(55, 395, "Line and step shapes illustrate model forms; the notebook reports fitted results.", 16, color=GRAY)
save("ml101_housing_workflow.svg", "Linear model versus decision tree", "Scaling is fitted on training rows for the linear model; the tree uses the unscaled training features.", ml101)

fashion = text(55, 121, "One 28 × 28 image becomes ten class scores in this CNN", 21, weight="bold")
fashion += rect(64, 157, 229, 190, fill="#e8eef1", stroke="#a8c2d1", radius=7)
fashion += '<path d="M110 227 L136 198 L170 210 L203 198 L230 227 L214 247 L201 239 L201 312 L140 312 L140 239 L126 247Z" fill="#526777"/>'
fashion += text(74, 372, "Input: grayscale clothing", 17, weight="bold")
fashion += line(301, 249, 359, 249)
fashion += text(372, 180, "Learned feature maps", 19, weight="bold")
for k, tint in enumerate([CYAN, BLUE, PINK, CYAN]):
    ox = 373 + k * 73
    fashion += rect(ox, 213, 69, 69, fill="#fff", stroke=tint, radius=3)
    for row in range(4):
        for col in range(4):
            fashion += f'<rect x="{ox+5+col*15}" y="{218+row*15}" width="13" height="13" fill="{tint}" opacity="{[.2,.45,.7,.9][(row+col+k)%4]}"/>'
fashion += text(372, 322, "Conv → ReLU → pool → Conv", 16, color=GRAY)
fashion += line(677, 249, 734, 249)
fashion += rect(745, 157, 399, 190, fill=LIGHT, stroke=PINK, radius=7)
fashion += text(765, 190, "Ten output scores", 19, weight="bold")
for j, (label, width) in enumerate([("T-shirt", 195), ("Dress", 104), ("Sneaker", 70), ("other classes", 42)]):
    y = 211 + j * 31
    fashion += text(765, y + 15, label, 15)
    fashion += rect(877, y, width, 18, fill=BLUE if j == 0 else "#a8c2d1", stroke="none", radius=3, width=0)
fashion += text(745, 372, "Compare predicted and known labels.", 16, color=GRAY)
fashion += text(745, 397, "Inspect held-out mistakes.", 16, color=GRAY)
save("fashion_classification_workflow.svg", "From Fashion-MNIST pixels to class scores", "The shirt and bars are schematic; the notebook plots actual images, loss, and predictions.", fashion)

image = lane(43, 99, 1114, 285)
image += text(70, 139, "Text-to-image lesson: make a controlled visual comparison", 20, weight="bold")
for j, (x, heading, detail, fill) in enumerate([
    (69, "Baseline", "short prompt", PALE),
    (369, "Add detail", "same model/settings", "#fff"),
    (669, "Change style", "same subject", PALE),
]):
    image += rect(x, 179, 250, 144, fill=fill)
    image += f'<path d="M{x+48} 262 L{x+112} 205 L{x+175} 262Z" fill="{[BLUE,PINK,CYAN][j]}"/>'
    image += f'<rect x="{x+63}" y="262" width="99" height="44" fill="#fff" stroke="{BLUE}"/>'
    image += text(x + 17, 352, heading + ": " + detail, 16, weight="bold")
    if j < 2:
        image += line(x + 257, 248, x + 291, 248)
image += text(970, 220, "Record", 19, weight="bold")
image += text(970, 252, "prompt", 16, color=GRAY)
image += text(970, 279, "model + seed", 16, color=GRAY)
image += text(970, 306, "saved image", 16, color=GRAY)
save("image_generation_learning_map.svg", "Prompt changes become visible image comparisons", "The sample cottages are schematic; interpret the images generated by the chosen service in the notebook.", image)

context = ""
for x, heading, prompt, labels, vals, tint in [
    (55, "Factual completion", "The capital of France is ...", ["Paris", "Lyon", "Rome"], [0.72, 0.17, 0.11], BLUE),
    (630, "Subjective completion", "The coolest city in China is ...", ["Shanghai", "Beijing", "Chengdu"], [0.38, 0.34, 0.28], PINK),
]:
    context += rect(x, 119, 515, 253, fill="#fff", stroke="#a8c2d1", radius=10)
    context += text(x + 20, 154, heading, 20, weight="bold")
    context += text(x + 20, 184, prompt, 17, color=GRAY)
    for j, (label, val) in enumerate(zip(labels, vals)):
        y = 217 + j * 45
        context += text(x + 20, y + 18, label, 16)
        context += rect(x + 142, y, int(330 * val), 23, fill=tint, stroke="none", radius=3, width=0)
        context += text(x + 482, y + 18, f"{val:.0%}", 15, color=GRAY, anchor="end")
save("next_token_context.svg", "Two prompts, different kinds of next-token question", "Candidate probabilities are invented examples; the notebook calculates actual GPT-2 top-k tokens for these prompts.", context)

classes = rect(55, 111, 1090, 277, fill=LIGHT, stroke=PINK, radius=20)
classes += text(80, 151, "One Person definition creates two independent objects", 21, weight="bold")
classes += rect(83, 177, 305, 161, fill="#fff", stroke=BLUE)
for j, s in enumerate(["class Person:", "  species = 'human'", "  __init__(name, age)", "  description()"]):
    classes += text(101, 211 + 30 * j, s, 18, color=BLUE if j == 0 else NAVY)
classes += line(394, 252, 474, 207)
classes += line(394, 252, 474, 302)
classes += rect(480, 171, 309, 86, fill=PALE, stroke=BLUE)
classes += text(502, 205, "person1 = Amina", 19, weight="bold")
classes += text(502, 235, "age: 3 → 4", 18, color=GRAY)
classes += rect(480, 277, 309, 86, fill="#fff", stroke=BLUE)
classes += text(502, 311, "person2 = Kai", 19, weight="bold")
classes += text(502, 341, "age: 5 (unchanged)", 18, color=GRAY)
classes += text(816, 216, "Shared class attribute", 19, weight="bold")
classes += text(816, 249, "Person.species", 18, color=BLUE)
classes += text(816, 279, "changes for both objects", 17, color=GRAY)
classes += text(816, 328, "Each age is stored separately.", 16, color=GRAY)
save("class_instance_map.svg", "Class, instances, and shared attributes", "The names and ages match the Person cells in this notebook.", classes)

data = text(55, 121, "Three separate examples in this lesson", 22, weight="bold")
data += text(55, 152, "The notebook changes sample data between sections; these are views, not one transformed dataset.", 16, color=GRAY)
for x, heading in [(55, "Python collection"), (435, "pandas table"), (815, "Matplotlib chart")]:
    data += rect(x, 178, 330, 194, fill="#fff", stroke="#a8c2d1", radius=10)
    data += text(x + 17, 213, heading, 19, weight="bold")
data += text(74, 254, 'person = {"name": "Alice",', 16, color=BLUE)
data += text(74, 282, '          "age": 22}', 16, color=BLUE)
data += text(74, 337, "Look up person['age']", 16, color=GRAY)
for y, a, b in [(253, "Alice", "25"), (283, "Bob", "30"), (313, "Charlie", "35")]:
    data += line(457, y + 6, 742, y + 6, color="#d4e5ed", width=1, arrow=False)
    data += text(457, y, a, 16)
    data += text(683, y, b, 16)
data += text(457, 237, "Name", 16, weight="bold")
data += text(683, 237, "Age", 16, weight="bold")
data += text(457, 352, "Read a row or column", 16, color=GRAY)
for j, (label, h) in enumerate(zip("ABCD", [28, 54, 70, 98])):
    x = 844 + j * 68
    data += rect(x, 330 - h, 43, h, fill=[BLUE, CYAN, PINK, BLUE][j], stroke="none", radius=2, width=0)
    data += text(x + 22, 353, label, 15, anchor="middle")
data += text(832, 236, "values = [23, 45, 56, 78]", 15, color=GRAY)
save("data_to_chart.svg", "Read the data used in each section", "The dictionary, DataFrame, and chart use different example values in the notebook.", data)

game = rect(55, 110, 1090, 276, fill=LIGHT, stroke=PINK, radius=20)
game += text(80, 151, "Amina's adventure makes variables, branches, and functions visible", 20, weight="bold")
game += rect(78, 178, 325, 168, fill="#fff", stroke=BLUE)
game += text(98, 214, "Hero state", 19, weight="bold")
game += text(98, 249, "name = Amina", 18)
game += text(98, 280, "inventory = map", 18)
game += text(98, 318, "These values change the story.", 16, color=GRAY)
game += rect(440, 178, 325, 168, fill="#fff", stroke=BLUE)
game += text(460, 214, "The if / elif branch", 19, weight="bold")
game += text(460, 249, "hero_health = 5", 18)
game += text(460, 282, '→ "You can enter,', 17, color=BLUE)
game += text(481, 310, 'but be careful."', 17, color=BLUE)
game += rect(802, 178, 316, 168, fill="#fff", stroke=BLUE)
game += text(822, 214, "A reusable function", 19, weight="bold")
game += text(822, 249, "drink_potion(7, 3)", 18)
game += text(822, 282, "→ 10 health points", 18, color=PINK)
game += text(822, 322, "A later cell resets health to 7.", 15, color=GRAY)
save("text_adventure_learning_map.svg", "A text adventure built from Python basics", "The branch and potion examples are separate cells with their own health values.", game)

control = text(55, 123, "Two experiments in the Week 8 notebook", 22, weight="bold")
control += rect(55, 154, 535, 224, fill="#fff", stroke=BLUE, radius=12)
control += text(75, 189, "A. Prompt wording and style", 20, weight="bold")
for j, (label, task) in enumerate([
    ("Direct", "robot learns to paint"),
    ("Example", "another robot wants to fly"),
    ("Writer role", "robot discovers a new skill"),
]):
    y = 219 + j * 43
    control += rect(76, y - 21, 110, 31, fill=PALE, stroke="none", radius=5, width=0)
    control += text(85, y, label, 16, weight="bold")
    control += text(201, y, task, 16)
control += text(76, 355, "Task wording changes here too.", 16, color=PINK)
control += rect(610, 154, 535, 224, fill="#fff", stroke=BLUE, radius=12)
control += text(630, 189, "B. Temperature with one fixed prompt", 20, weight="bold")
control += text(630, 223, '"robot learning to paint"', 18, color=GRAY)
for j, (label, tint) in enumerate([("T = 0.0", BLUE), ("T = 0.1", CYAN), ("T = 1.0", PINK)]):
    x = 635 + j * 158
    control += rect(x, 251, 135, 56, fill=PALE, stroke=tint, radius=7)
    control += text(x + 67, 285, label, 18, anchor="middle", font="Menlo,monospace")
control += text(630, 355, "Three outputs per setting; inspect variation.", 16, color=GRAY)
save("llm_output_control_map.svg", "Compare the two kinds of LLM output experiment", "Prompt wording varies in Part A; only temperature changes across Part B's repeated fixed prompt.", control)

shap_map = lane(43, 99, 1114, 283)
shap_map += text(68, 139, "Train a predictor, then ask two different explanation questions", 20, weight="bold")
shap_map += card(70, 185, 238, 112, "Housing rows", "features + target", fill="#fff")
shap_map += line(312, 240, 373, 240)
shap_map += card(379, 185, 259, 112, "Fitted model", "predicts held-out values", fill=PALE)
shap_map += line(642, 240, 700, 197)
shap_map += line(642, 240, 700, 286)
shap_map += card(708, 154, 385, 88, "Global view", "patterns across selected rows", fill="#fff")
shap_map += card(708, 270, 385, 88, "Local view", "contributions for one prediction", fill="#fff")
save("shap_learning_map.svg", "Prediction first; explanation second", "SHAP describes the fitted model on chosen rows, not a causal account of housing prices.", shap_map)

llm = text(55, 121, "Three questions asked of one LLM in this notebook", 22, weight="bold")
for x, heading, prompt, observe in [
    (55, "Ask", '"How can I save energy at home?"', "Read the answer; check the advice."),
    (435, "Sample", '"A robot learning to paint..."', "Run it three times at each T."),
    (815, "Guide", '"Label this SUPPORT / CONCERN"', "Add examples; check the label."),
]:
    llm += rect(x, 155, 330, 193, fill="#fff", stroke=BLUE, radius=14)
    llm += text(x + 18, 190, heading, 21, weight="bold")
    llm += rect(x + 18, 207, 293, 58, fill=PALE, stroke="none", radius=9, width=0)
    llm += text(x + 28, 241, prompt, 16)
    llm += text(x + 18, 306, observe, 16, color=GRAY)
llm += text(55, 388, "Changing temperature and adding examples are different experiments; neither guarantees a true answer.", 16, color=GRAY)
save("llm_basics_learning_map.svg", "Ask, sample, guide", "The prompts shown correspond to the lesson's energy, robot story, and sentence-label cells.", llm)

print("Generated teaching figures in", OUT)
