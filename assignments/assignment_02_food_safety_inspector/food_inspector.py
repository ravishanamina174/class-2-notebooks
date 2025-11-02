"""
Assignment 2: AI Food Safety Inspector
Zero-Shot Prompting with Structured Outputs

Your mission: Analyze restaurant reviews and complaints to detect health violations
using only clear instructions — no training examples needed!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()


class ViolationCategory(Enum):
    TEMPERATURE_CONTROL = "Food Temperature Control"
    PERSONAL_HYGIENE = "Personal Hygiene"
    PEST_CONTROL = "Pest Control"
    CROSS_CONTAMINATION = "Cross Contamination"
    FACILITY_MAINTENANCE = "Facility Maintenance"
    UNKNOWN = "Unknown"


class SeverityLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class InspectionPriority(Enum):
    URGENT = "URGENT"
    HIGH = "HIGH"
    ROUTINE = "ROUTINE"
    LOW = "LOW"


@dataclass
class Violation:
    """Structured violation data"""

    category: str
    description: str
    severity: str
    evidence: str
    confidence: float


@dataclass
class InspectionReport:
    """Complete inspection analysis"""

    restaurant_name: str
    overall_risk_score: int
    violations: List[Violation]
    inspection_priority: str
    recommended_actions: List[str]
    follow_up_required: bool


class FoodSafetyInspector:
    """
    AI-powered food safety analyzer using zero-shot structured prompting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize with LLM for consistent violation detection."""
        # TODO: Initialize an LLM for consistent JSON-style outputs
        # self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.analysis_chain = None
        self.risk_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot prompts for violation detection and risk assessment.

        Create TWO chains:
        1. analysis_chain: Detects violations and extracts evidence
        2. risk_chain: Calculates risk scores based on violations

        Requirements:
        - Must output valid JSON
        - Include all violation categories
        - Extract specific evidence quotes
        - Generate confidence scores
        """

        # TODO: Create violation detection prompt (as a raw template string)
        analysis_template_str = (
            "You are a food safety inspector AI. Analyze the review below for food safety violations.\n"
            "Return ONLY valid JSON.\n\n"
            "Rules:\n"
            "- Detect violations from these categories:\n"
            "  TEMPERATURE_CONTROL, PERSONAL_HYGIENE, PEST_CONTROL,\n"
            "  CROSS_CONTAMINATION, FACILITY_MAINTENANCE, UNKNOWN\n"
            "- Extract EXACT evidence quotes from the text\n"
            "- For each violation include: category, description, severity, evidence, confidence (0-1)\n"
            "- If no violations, return empty list\n\n"
            "JSON format:\n"
            "{\n"
            '  "violations": [\n'
            "    {\n"
            '      "category": "...",\n'
            '      "description": "...",\n'
            '      "severity": "...",\n'
            '      "evidence": "...",\n'
            '      "confidence": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Text:\n{review_text}\n"
        )

        # TODO: Create risk assessment prompt (as a raw template string)
        risk_template_str = (
            "You are a food safety inspector. Based on violations below, assign a risk score 0-100.\n"
            "Rules:\n"
            "- Critical = +40\n"
            "- High = +25\n"
            "- Medium = +15\n"
            "- Low = +5\n"
            "- Cap at 100\n\n"
            "Return ONLY JSON:\n"
            '{ "risk_score": number }\n\n'
            "Violations:\n{violations}"
        )

        # TODO: Build PromptTemplate objects from the strings above
        analysis_template = PromptTemplate.from_template(analysis_template_str)
        risk_template = PromptTemplate.from_template(risk_template_str)

        # TODO: Set up the chains
        # self.analysis_chain = analysis_template | self.llm
        # self.risk_chain = risk_template | self.llm
        self.analysis_chain = analysis_template | self.llm | StrOutputParser()
        self.risk_chain = risk_template | self.llm | StrOutputParser()

    def detect_violations(self, text: str) -> List[Violation]:
        """
        TODO #2: Detect health violations from text input.

        Args:
            text: Review, complaint, or social media post

        Returns:
            List of Violation objects with evidence
        """

        # TODO: Use analysis_chain to detect violations
        # try:
        #     raw_response = self.analysis_chain.invoke({"review_text": text})
        #     data = json.loads(raw_response)
        #     violations: List[Violation] = [...]
        #     return violations
        # except Exception as e:
        #     print(f"Error detecting violations: {e}")
        #     return []
        try:
            raw = self.analysis_chain.invoke({"review_text": text})
            data = json.loads(raw)

            violations = []
            for v in data.get("violations", []):
                violations.append(
                    Violation(
                        category=v.get("category", "Unknown"),
                        description=v.get("description", ""),
                        severity=v.get("severity", "Low"),
                        evidence=v.get("evidence", ""),
                        confidence=float(v.get("confidence", 0))
                    )
                )
            return violations
        except Exception as e:
            print("Error in detect_violations:", e)
            return []

    def calculate_risk_score(self, violations: List[Violation]) -> Tuple[int, str]:
        """
        TODO #3: Calculate overall risk score and determine inspection priority.

        Args:
            violations: List of detected violations

        Returns:
            Tuple of (risk_score, inspection_priority)
        """

        # TODO: Implement risk scoring logic
        # Consider: severity levels, number of violations, categories affected

        # risk_score = ...
        # priority = ...
        # return risk_score, priority
        severity_weights = {
            "Critical": 40,
            "High": 25,
            "Medium": 15,
            "Low": 5
        }

        score = 0
        for v in violations:
            score += severity_weights.get(v.severity, 5)

        score = min(score, 100)

        if score >= 70:
            priority = "URGENT"
        elif score >= 50:
            priority = "HIGH"
        elif score >= 25:
            priority = "ROUTINE"
        else:
            priority = "LOW"

        return score, priority

    def analyze_review(
        self, text: str, restaurant_name: str = "Unknown"
    ) -> InspectionReport:
        """
        TODO #4: Complete analysis pipeline for a single review.

        Args:
            text: Review text to analyze
            restaurant_name: Name of the restaurant

        Returns:
            Complete InspectionReport with all findings
        """

        # TODO: Implement full analysis pipeline
        # 1. Detect violations
        # 2. Calculate risk score
        # 3. Generate recommendations
        # 4. Create InspectionReport

        # violations = self.detect_violations(text)
        # risk_score, priority = self.calculate_risk_score(violations)
        # recommendations = [...]
        # return InspectionReport(...)
        violations = self.detect_violations(text)
        violations = self.filter_false_positives(violations)

        risk_score, priority = self.calculate_risk_score(violations)

        recommendations = []
        for v in violations:
            if "Pest" in v.category:
                recommendations.append("Schedule pest control")
            if "Hygiene" in v.category:
                recommendations.append("Enforce proper hygiene training")
            if "Temperature" in v.category:
                recommendations.append("Check refrigeration and food heating procedures")

        follow_up = risk_score >= 50 or len(violations) > 0

        return InspectionReport(
            restaurant_name=restaurant_name,
            overall_risk_score=risk_score,
            violations=violations,
            inspection_priority=priority,
            recommended_actions=list(set(recommendations)),
            follow_up_required=follow_up,
        )

    def batch_analyze(self, reviews: List[Dict[str, str]]) -> InspectionReport:
        """
        TODO #5: Analyze multiple reviews for the same restaurant.

        Args:
            reviews: List of dicts with 'text' and 'source' keys

        Returns:
            Aggregated InspectionReport
        """

        # TODO: Implement aggregation logic
        # - Combine violations from multiple sources
        # - Weight by source reliability
        # - Remove duplicates
        # - Calculate aggregate risk score

        # aggregated_report = ...
        # return aggregated_report
        all_violations = []
        for r in reviews:
            all_violations.extend(self.detect_violations(r["text"]))

        all_violations = self.filter_false_positives(all_violations)

        score, priority = self.calculate_risk_score(all_violations)

        return InspectionReport(
            restaurant_name="Multiple Reviews",
            overall_risk_score=score,
            violations=all_violations,
            inspection_priority=priority,
            recommended_actions=[],
            follow_up_required=score >= 50,
        )

    def filter_false_positives(self, violations: List[Violation]) -> List[Violation]:
        """
        TODO #6 (Bonus): Filter out likely false positives.

        Consider:
        - Sarcasm indicators
        - Exaggeration patterns
        - Confidence thresholds
        """

        # TODO: Implement false positive filtering
        # return filtered
        filtered = []
        for v in violations:
            if v.confidence < 0.4:
                continue
            if "just kidding" in v.evidence.lower():
                continue
            filtered.append(v)
        return filtered



def test_inspector():
    """Test the food safety inspector with various scenarios."""

    inspector = FoodSafetyInspector()

    # Test cases with varying violation types
    test_reviews = [
        {
            "restaurant": "Bob's Burgers",
            "text": "Great food but saw a mouse run across the dining room! Also, the chef wasn't wearing gloves while handling raw chicken.",
        },
        {
            "restaurant": "Pizza Palace",
            "text": "Just left and the bathroom had no soap, and I'm pretty sure that meat sitting on the counter wasn't refrigerated 😷",
        },
        {
            "restaurant": "Sushi Express",
            "text": "Love this place! Though it's weird they keep the raw fish next to the vegetables #sushitime #questionable",
        },
        {
            "restaurant": "Taco Town",
            "text": "Best tacos in town! Super clean kitchen, staff always wears hairnets, everything looks fresh!",
        },
        {
            "restaurant": "Burger Barn",
            "text": "The cockroach in my salad added extra protein! Just kidding, but seriously the place needs cleaning.",
        },
    ]

    print("🍽️ FOOD SAFETY INSPECTION SYSTEM 🍽️\n")
    print("=" * 70)

    for review_data in test_reviews:
        print(f"\n🏪 Restaurant: {review_data['restaurant']}")
        print(f"📝 Review: \"{review_data['text'][:100]}...\"")

        # Analyze the review
        report = inspector.analyze_review(
            review_data["text"], review_data["restaurant"]
        )

        # Display results
        print(f"\n📊 Inspection Report:")
        print(f"  Risk Score: {report.overall_risk_score}/100")
        print(f"  Priority: {report.inspection_priority}")
        print(f"  Violations Found: {len(report.violations)}")

        if report.violations:
            print("\n  Detected Violations:")
            for v in report.violations:
                print(f"    • [{v.severity}] {v.category}: {v.description}")
                print(f'      Evidence: "{v.evidence[:50]}..."')
                print(f"      Confidence: {v.confidence:.0%}")

        if report.recommended_actions:
            print("\n  Recommended Actions:")
            for action in report.recommended_actions:
                print(f"    ✓ {action}")

        print(f"\n  Follow-up Required: {'Yes' if report.follow_up_required else 'No'}")
        print("-" * 70)

    # Test batch analysis
    print("\n🔬 BATCH ANALYSIS TEST:")
    print("=" * 70)

    # Multiple reviews for same restaurant
    batch_reviews = [
        {"text": "Saw bugs in the kitchen!", "source": "Yelp"},
        {"text": "Food was cold and undercooked", "source": "Google"},
        {"text": "Staff not wearing hairnets", "source": "Twitter"},
    ]

    
    batch_report = inspector.batch_analyze(batch_reviews)
    print(f"Aggregate Risk Score: {batch_report.overall_risk_score}/100")
    print(f"Total Violations: {len(batch_report.violations)}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    test_inspector()
