from experitur.util import cprint
import termcolor

for attr in [None] + list(termcolor.ATTRIBUTES.keys()):
    for highlight in [None] + list(termcolor.HIGHLIGHTS.keys()):
        for color in [None] + list(termcolor.COLORS.keys()):
            cprint(
                f"{attr} {highlight} {color}",
                color=color,
                on_color=highlight,
                attrs=[attr] if attr is not None else None,
                end="",
            )
            print()
