"""
Parametric fallback template: "What is the difference between ionic and covalent bonding?"

Pre-tested and deterministic. Used when LLM retries are exhausted.
"""

TOPIC = "ionic_vs_covalent"
TITLE = "Ionic vs. Covalent Bonding"

NARRATION_SCRIPT = """
Chemical bonds hold atoms together, but not all bonds are the same.
The two most common types are ionic bonds and covalent bonds.

In an ionic bond, one atom completely transfers one or more electrons to another.
This creates two oppositely charged particles called ions. Positive ions and
negative ions attract each other strongly. Table salt, sodium chloride, is
a perfect example. Sodium gives its electron to chlorine, creating
Na-plus and Cl-minus ions that lock together in a crystal lattice.

In a covalent bond, atoms do not transfer electrons — they share them.
Both atoms hold onto the shared electrons, achieving a stable outer shell
without becoming ions. Water and carbon dioxide are covalent molecules.

The key difference is transfer versus sharing. Ionic bonding involves
complete electron transfer between a metal and a non-metal, producing
an ionic compound with high melting points that conducts electricity when
dissolved. Covalent bonding involves electron sharing between non-metals,
forming molecules with lower melting points that usually do not conduct electricity.
"""

ANIMATION_CODE = '''
from manim import *

class ChemistryScene(Scene):
    def construct(self):
        title = Text("Ionic vs. Covalent Bonding", font_size=40, color=WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # ── IONIC ────────────────────────────────────────────────────────────
        ionic_title = Text("Ionic Bond: Electron Transfer", font_size=32, color=ORANGE).shift(UP * 3)
        self.play(Write(ionic_title))

        na = Circle(radius=0.6, color=YELLOW, fill_opacity=0.9).shift(LEFT * 2.5)
        na_label = MathTex("Na", color=BLACK).move_to(na.get_center())
        na_e = Dot(color=RED, radius=0.12).move_to(na.get_right() + RIGHT * 0.15)

        cl = Circle(radius=0.6, color=GREEN, fill_opacity=0.9).shift(RIGHT * 2.5)
        cl_label = MathTex("Cl", color=BLACK).move_to(cl.get_center())

        self.play(FadeIn(na, na_label, na_e, cl, cl_label))
        self.wait(1)

        # Transfer electron
        self.play(na_e.animate.move_to(cl.get_left() + LEFT * 0.15), run_time=1.5)
        self.wait(0.5)

        na_plus = MathTex(r"Na^+", color=YELLOW, font_size=48).shift(LEFT * 2.5)
        cl_minus = MathTex(r"Cl^-", color=GREEN, font_size=48).shift(RIGHT * 2.5)
        ionic_note = Text("Complete transfer → ionic bond", font_size=22, color=ORANGE).shift(DOWN * 1.5)
        self.play(Transform(na_label, na_plus), Transform(cl_label, cl_minus), Write(ionic_note))
        self.wait(2)

        # ── TRANSITION ───────────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # ── COVALENT ─────────────────────────────────────────────────────────
        cov_title = Text("Covalent Bond: Electron Sharing", font_size=32, color=BLUE).shift(UP * 3)
        self.play(Write(cov_title))

        h1 = Circle(radius=0.5, color=BLUE, fill_opacity=0.8).shift(LEFT * 1.5)
        h1_l = MathTex("H", color=WHITE).move_to(h1.get_center())
        h2 = Circle(radius=0.5, color=BLUE, fill_opacity=0.8).shift(RIGHT * 1.5)
        h2_l = MathTex("H", color=WHITE).move_to(h2.get_center())

        e1 = Dot(color=YELLOW).move_to(h1.get_center())
        e2 = Dot(color=YELLOW).move_to(h2.get_center())

        self.play(FadeIn(h1, h1_l, h2, h2_l))
        self.wait(0.5)
        self.play(e1.animate.move_to(ORIGIN + LEFT * 0.3), e2.animate.move_to(ORIGIN + RIGHT * 0.3))
        bond = Line(LEFT * 0.4, RIGHT * 0.4, color=YELLOW, stroke_width=5)
        cov_note = Text("Shared electrons → covalent bond", font_size=22, color=BLUE).shift(DOWN * 1.5)
        self.play(Create(bond), Write(cov_note))
        self.wait(2)

        # ── COMPARISON TABLE ─────────────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        table_title = Text("Key Differences", font_size=36, color=WHITE).shift(UP * 3)
        self.play(Write(table_title))

        rows = [
            ("Property",       "Ionic",           "Covalent"),
            ("Electron",       "Transfer",        "Sharing"),
            ("Atoms",          "Metal + Non-metal","Non-metal"),
            ("Melting Point",  "High",            "Lower"),
            ("Conducts?",      "Yes (dissolved)", "Usually No"),
        ]
        start_y = 1.8
        for i, (prop, ionic, cov) in enumerate(rows):
            colour = YELLOW if i == 0 else WHITE
            row = Text(f"{prop:<16} {ionic:<22} {cov}", font_size=20, color=colour,
                       font="Courier New").shift(UP * (start_y - i * 0.7))
            self.play(Write(row), run_time=0.4)

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        outro = Text("Ionic: transfer · Covalent: sharing", font_size=34, color=WHITE)
        self.play(Write(outro))
        self.wait(2)
'''
