"""
Parametric fallback template: "Why do atoms form covalent bonds?"

Pre-tested and deterministic. Used when LLM retries are exhausted.
"""

TOPIC = "covalent_bonding"
TITLE = "Why Do Atoms Form Covalent Bonds?"

NARRATION_SCRIPT = """
Atoms form covalent bonds to become more stable. Every atom wants to have a
full outer shell of electrons. When two atoms share electrons, both can achieve
that stable configuration.
Take hydrogen as the simplest example. Each hydrogen atom has one electron
and needs two to fill its outer shell. When two hydrogen atoms come together,
they share their electrons. Both atoms now feel as if they have a full shell.
This sharing of electrons is called a covalent bond.
Water is another great example. Oxygen needs two more electrons to be stable.
Each of the two hydrogen atoms shares its electron with oxygen, forming two
covalent bonds. The result is the water molecule H-two-O.
Covalent bonds are strong and directional, which is why molecules have
specific shapes. The shape determines how molecules interact with each other
and with our bodies.
"""

ANIMATION_CODE = '''
from manim import *

class ChemistryScene(Scene):
    def construct(self):
        title = Text("Covalent Bonds: Sharing Electrons", font_size=40, color=WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Two hydrogen atoms approaching
        h1 = Circle(radius=0.5, color=BLUE, fill_opacity=0.8).shift(LEFT * 3)
        h1_label = MathTex("H", color=WHITE).move_to(h1.get_center())
        h1_e = Dot(color=YELLOW).move_to(h1.get_right())

        h2 = Circle(radius=0.5, color=BLUE, fill_opacity=0.8).shift(RIGHT * 3)
        h2_label = MathTex("H", color=WHITE).move_to(h2.get_center())
        h2_e = Dot(color=YELLOW).move_to(h2.get_left())

        self.play(FadeIn(h1, h1_label, h1_e), FadeIn(h2, h2_label, h2_e))
        self.wait(1)

        bond_label = Text("Atoms need full outer shells", font_size=24, color=YELLOW).shift(UP * 2.5)
        self.play(Write(bond_label))
        self.wait(1)

        # Move together
        self.play(
            h1.animate.shift(RIGHT * 2),
            h1_label.animate.shift(RIGHT * 2),
            h1_e.animate.shift(RIGHT * 2),
            h2.animate.shift(LEFT * 2),
            h2_label.animate.shift(LEFT * 2),
            h2_e.animate.shift(LEFT * 2),
        )
        self.wait(0.5)

        bond_line = Line(LEFT * 0.3, RIGHT * 0.3, color=YELLOW, stroke_width=4)
        shared = Text("Shared electrons → Covalent Bond", font_size=24, color=GREEN).shift(DOWN * 1.8)
        self.play(Create(bond_line), Write(shared))
        self.wait(2)

        # Water molecule
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        water_title = Text("Example: Water (H₂O)", font_size=36, color=WHITE)
        self.play(Write(water_title))
        self.wait(1)
        self.play(FadeOut(water_title))

        o = Circle(radius=0.6, color=RED, fill_opacity=0.9)
        o_label = MathTex("O", color=WHITE).move_to(o.get_center())
        h_left = Circle(radius=0.4, color=BLUE, fill_opacity=0.8).shift(LEFT * 1.5 + DOWN * 1)
        h_left_l = MathTex("H", color=WHITE).move_to(h_left.get_center())
        h_right = Circle(radius=0.4, color=BLUE, fill_opacity=0.8).shift(RIGHT * 1.5 + DOWN * 1)
        h_right_l = MathTex("H", color=WHITE).move_to(h_right.get_center())
        bond1 = Line(o.get_bottom(), h_left.get_top(), color=YELLOW, stroke_width=3)
        bond2 = Line(o.get_bottom(), h_right.get_top(), color=YELLOW, stroke_width=3)

        self.play(
            FadeIn(o, o_label, h_left, h_left_l, h_right, h_right_l),
            Create(bond1), Create(bond2)
        )
        formula = MathTex(r"H_2O", font_size=60, color=YELLOW).shift(UP * 2.5)
        self.play(Write(formula))
        self.wait(3)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        outro = Text("Covalent bonds: stability through sharing.", font_size=34, color=WHITE)
        self.play(Write(outro))
        self.wait(2)
'''
