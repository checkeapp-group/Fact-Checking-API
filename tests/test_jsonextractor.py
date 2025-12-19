import json
import unittest

from veridika.src.llm.utils import extract_json

# --------------------------------------------------------------------------- #
#  Test vectors                                                               #
# --------------------------------------------------------------------------- #
OBJ: dict[str, list[str]] = {  # reusable baseline payload
    "búsquedas": ["python", "json"],
    "preguntas": ["¿qué es?", "¿cómo funciona?"],
}


def to_expect() -> dict[str, list[str]]:
    """Return a *fresh* copy every time (tests may mutate)."""
    return {k: v.copy() for k, v in OBJ.items()}


# --------------------------------------------------------------------------- #
#  Test case                                                                  #
# --------------------------------------------------------------------------- #
class ExtractJsonTests(unittest.TestCase):
    # 1. simplest possible hit ------------------------------------------------
    def test_basic_object(self):
        resp = json.dumps(OBJ)
        self.assertEqual(extract_json(resp), to_expect())

    # 2. fenced code + trailing commas ---------------------------------------
    def test_code_fence_and_trailing_commas(self):
        fenced = '```json\n{\n  "búsquedas": ["a","b",],\n  "preguntas": ["c","d",]\n}\n```'
        expected = {"búsquedas": ["a", "b"], "preguntas": ["c", "d"]}
        self.assertEqual(extract_json(fenced), expected)

    # 3. multiple JSON fragments; largest wins --------------------------------
    def test_multiple_json_pick_largest(self):
        small = '{"foo": "bar"}'
        large = json.dumps(OBJ)
        resp = f"Irrelevant text {small} more text {large} end."
        self.assertEqual(extract_json(resp), to_expect())

    # 4. array of dicts should be merged --------------------------------------
    def test_array_of_dicts_is_merged(self):
        arr = '[{"búsquedas": ["one"]}, {"preguntas": ["two"]}]'
        expected = {"búsquedas": ["one"], "preguntas": ["two"]}
        self.assertEqual(extract_json(arr), expected)

    # 5. missing field(s) ⇒ None ---------------------------------------------
    def test_missing_fields_returns_none(self):
        incomplete = '{"búsquedas": ["only searches"]}'
        self.assertIsNone(extract_json(incomplete))

    # 6. case‑insensitive field matching --------------------------------------
    def test_field_name_case_insensitive(self):
        mixed_case = '{"BÚsquedas": ["x"], "Preguntas": ["y"]}'
        expected = {"búsquedas": ["x"], "preguntas": ["y"]}
        self.assertEqual(extract_json(mixed_case), expected)

    def test_single_quotes(self):
        single_quotes = "{'búsquedas': ['x'], 'preguntas': ['y']}"
        expected = {"búsquedas": ["x"], "preguntas": ["y"]}
        self.assertEqual(extract_json(single_quotes), expected)

    def test_between_text(self):
        text = """
Did you ever hear the tragedy of Darth Plagueis The Wise?I thought not. 
It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith, 
so powerful and so wise he could use the Force to influence the midichlorians to create life… He had such a knowledge 

{
    "búsquedas": ["x","x"],
    "preguntas": ["y", "w"]
}
of the dark side, he could even keep the ones he cared about from dying.The dark side of the Force 
is a pathway to many abilities some consider to be unnatural.He became so powerful… the only thing he was afraid of was 
losing his power, which eventually, of course, he did. Unfortunately, he taught his apprentice everything he knew, then his 
apprentice killed him in his sleep. Ironic. He could save others from death, but not himself.
"""

        expected = {"búsquedas": ["x", "x"], "preguntas": ["y", "w"]}
        self.assertEqual(extract_json(text), expected)

    def test_between_text2(self):
        text = """
Did you ever hear the tragedy of Darth Plagueis The Wise?I thought not. {"Fake": "json"}
It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith, 
so powerful and so wise he could use the Force to influence the midichlorians to create life… He had such a knowledge 
{}

{
    "búsquedas": ["x","x"],
    "preguntas": ["y", "w"]
}
of the dark side, he could even keep the ones he cared about from dying.The dark side of the Force 
is a pathway to many abilities some {hello} consider to be unnatural.He became so powerful… the only thing he was afraid of was 
losing his power, which eventually, of course, he did. Unfortunately, he taught his apprentice everything he knew, then his 
apprentice killed him in his sleep. Ironic. He could save others from death, but not himself.
"""

        expected = {"búsquedas": ["x", "x"], "preguntas": ["y", "w"]}

        self.assertEqual(extract_json(text), expected)

    def test_between_text3(self):
        text = """
Did you ever hear the tragedy of Darth Plagueis The Wise?I thought not. {"Fake": "json"}
It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith, 
so powerful and so wise he could use the Force to influence the midichlorians to create life… He had such a knowledge 
{}
{
'búsquedas': ['w'], 'preguntas': ['l']


}

of the dark side, he could even keep the ones he cared about from dying.The dark side of the Force 
is a pathway to many abilities some {hello} consider to be unnatural.He became so powerful… the only thing he was afraid of was 
losing his power, which eventually, of course, he did. Unfortunately, he taught his apprentice everything he knew, then his 
apprentice killed him in his sleep. Ironic. He could save others from death, but not himself.
"""

        expected = {"búsquedas": ["w"], "preguntas": ["l"]}

        self.assertEqual(extract_json(text), expected)

    def test_between_text_with_trailing_commas(self):
        text = """
        ```json
{
    "búsquedas": ["x","x"],
    "preguntas": ["y", "w"]
}
```
"""
        expected = {"búsquedas": ["x", "x"], "preguntas": ["y", "w"]}
        self.assertEqual(extract_json(text), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
