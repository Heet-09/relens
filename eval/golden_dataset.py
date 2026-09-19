"""Golden regression cases for the 7 example query types in the assessment brief, plus
edge cases. Ground truth is pulled directly from the structured records the app itself
extracted from these 4 brochures — re-verify against data/structured/*.json if you
re-ingest with a different model/prompt and these start failing.

Requires these 4 projects ingested before running: Lodha Regalia, Golden Willows at
Hiranandani Fortune City, Ashwini Elite, Codename Connexa.

Each case:
  question            the exact query sent to POST /query
  expected_intent      (optional) intent the router must return
  must_contain          (optional) substrings that must appear (case-insensitive) in
                         answer_text + table_data combined
  must_contain_one_of   (optional) at least one of these substrings must appear
  must_not_contain      (optional) substrings that must NOT appear (catches
                         hallucination / cross-project attribution leaks)
  must_have_table        (optional) table_data must be non-empty
  manual_review          (optional) too subjective for keyword matching; the runner
                          just prints the answer instead of asserting pass/fail
"""

NOT_FOUND_PHRASES = ["not found", "not stated", "not mentioned", "not specified", "no information"]

GOLDEN_CASES = [
    # 1. Factual Q&A
    {
        "question": "Where is Codename Connexa located and what is the expected handover date?",
        "expected_intent": "factual_qa",
        "must_contain": ["mahim west"],
        "must_contain_one_of": NOT_FOUND_PHRASES,  # handover date is null in the source
    },
    {
        "question": "Where is Ashwini Elite located?",
        "expected_intent": "factual_qa",
        "must_contain": ["tilak nagar"],
    },
    # 2. Amenity Filter
    {
        "question": "List all projects that include a swimming pool and a gym.",
        "expected_intent": "amenity_filter",
        "must_contain": ["lodha regalia", "golden willows"],
        "must_not_contain": ["ashwini elite", "codename connexa"],
    },
    {
        "question": "Which project has EV charging stations?",
        "expected_intent": "amenity_filter",
        "must_contain": ["codename connexa"],
        "must_not_contain": ["lodha regalia", "golden willows", "ashwini elite"],
    },
    # 3. Unit Type Filter
    {
        "question": "Which projects offer 3-bedroom units with a maid's room?",
        "expected_intent": "unit_filter",
        "manual_review": True,  # correct behavior is nuanced: Ashwini has a 3BHK but no
        # confirmed maid's room — a good answer either excludes it or flags it as unconfirmed,
        # neither of which is a clean keyword check
    },
    {
        "question": "Which projects have shop units under 200 sq ft?",
        "expected_intent": "unit_filter",
        "must_contain": ["ashwini elite"],
    },
    # 4. Cross-Document Comparison
    {
        "question": "Compare Ashwini Elite and Codename Connexa on location, amenities, and unit types.",
        "expected_intent": "comparison",
        "must_contain": ["ashwini elite", "codename connexa"],
        "must_have_table": True,  # regression guard: this exact case previously came back
        # with an empty table because the comparison call's max_tokens was too small
    },
    {
        "question": "Compare Lodha Regalia and Golden Willows on amenities.",
        "expected_intent": "comparison",
        "must_contain": ["lodha regalia", "golden willows"],
        "must_have_table": True,
    },
    # 5. Specification Lookup
    {
        "question": "What is the total area of the 2 BHK - Large unit in Codename Connexa?",
        "expected_intent": "factual_qa",
        "must_contain": ["738.62"],
    },
    {
        "question": "What is the total area of a 3-bedroom unit in Golden Willows?",
        "expected_intent": "factual_qa",
        "must_contain_one_of": NOT_FOUND_PHRASES,  # Golden Willows has no unit_types extracted
    },
    # 6. Developer / Project Info
    {
        "question": "Who is the developer of Lodha Regalia and when is the expected handover?",
        "expected_intent": "factual_qa",
        "must_contain": ["lodha"],
    },
    {
        "question": "Who is the developer of Ashwini Elite?",
        "expected_intent": "factual_qa",
        "must_contain": ["elite group"],
    },
    # 7. Summarisation
    {
        "question": "Give me a one-paragraph summary of each uploaded project.",
        "expected_intent": "summarization",
        "must_contain": ["lodha regalia", "golden willows", "ashwini elite", "codename connexa"],
    },
    # Edge cases
    {
        "question": "What is the swimming pool depth at Codename Connexa?",
        "must_contain_one_of": NOT_FOUND_PHRASES,  # Connexa has no pool at all
    },
    {
        "question": "What is the price per square foot for Ashwini Elite?",
        "must_contain_one_of": NOT_FOUND_PHRASES,  # no pricing was extracted from any brochure
    },
    {
        "question": "What is the RERA number for Ashwini Elite?",
        "must_contain": ["p51800024954"],
        # attribution check: must not leak one of Golden Willows' 8 building-wise RERA numbers
        "must_not_contain": ["p52000050193", "p52000050196", "p52000054617"],
    },
]
