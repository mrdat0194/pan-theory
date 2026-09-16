"""
Parametric fallback template: "How does the pH scale work?"

This scene is pre-tested and deterministic — it is used when the LLM
self-correction loop is exhausted to guarantee the learner receives a
video rather than a failure.
"""

TOPIC = "ph_scale"
TITLE = "How Does the pH Scale Work?"

NARRATION_SCRIPT = """
The pH scale is a way to measure how acidic or basic a solution is.
It runs from zero to fourteen. A pH of seven is neutral — that is pure water.
Values below seven are acidic. The lower the number, the stronger the acid.
Lemon juice, for example, has a pH of about two.
Values above seven are basic, or alkaline. Baking soda has a pH of around nine.
The scale is logarithmic, which means each whole-number step represents a
tenfold change in acidity. So a solution at pH three is ten times more
acidic than one at pH four, and one hundred times more acidic than pH five.
Understanding pH is important in chemistry, biology, and everyday life —
from digestion in our stomachs to the water in a swimming pool.
"""

ANIMATION_CODE = '''
from manim import *

class ChemistryScene(Scene):
    def construct(self):
        # Title
        title = Text("The pH Scale", font_size=48, color=WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Draw the pH bar
        bar = Rectangle(width=10, height=0.8, fill_opacity=1)
        bar.set_color(color=[RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE])
        bar_label = Text("0                    7                   14",
                         font_size=20).next_to(bar, DOWN, buff=0.2)

        self.play(FadeIn(bar), Write(bar_label))

        # Labels
        acid_label = Text("Acidic", font_size=28, color=RED).shift(LEFT * 3.5 + UP * 1.2)
        neutral_label = Text("Neutral", font_size=28, color=GREEN).shift(UP * 1.2)
        base_label = Text("Basic", font_size=28, color=BLUE).shift(RIGHT * 3.5 + UP * 1.2)

        self.play(Write(acid_label), Write(neutral_label), Write(base_label))
        self.wait(2)

        # Water marker
        marker = Triangle(color=YELLOW, fill_opacity=1).scale(0.2)
        water = Text("Water (pH 7)", font_size=22, color=YELLOW).shift(DOWN * 1.5)
        self.play(FadeIn(marker), Write(water))
        self.wait(1)

        # Lemon juice
        lemon_dot = Dot(color=ORANGE).shift(LEFT * 4 + DOWN * 0.4)
        lemon_label = Text("Lemon Juice (~pH 2)", font_size=20, color=ORANGE).shift(LEFT * 3 + DOWN * 1.5)
        self.play(FadeIn(lemon_dot), Write(lemon_label))
        self.wait(1)

        # Baking soda
        baking_dot = Dot(color=BLUE).shift(RIGHT * 1.6 + DOWN * 0.4)
        baking_label = Text("Baking Soda (~pH 9)", font_size=20, color=BLUE).shift(RIGHT * 3 + DOWN * 1.8)
        self.play(FadeIn(baking_dot), Write(baking_label))
        self.wait(2)

        # Logarithmic note
        log_note = Text("Each step = 10× change in acidity", font_size=24,
                        color=YELLOW).shift(DOWN * 2.6)
        self.play(Write(log_note))
        self.wait(3)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        outro = Text("pH: The Scale of Acids & Bases", font_size=36, color=WHITE)
        self.play(Write(outro))
        self.wait(2)
'''
