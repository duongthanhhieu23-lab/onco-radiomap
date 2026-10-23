"""Project-specific data contracts for Onco-RadioMap.

The biomedical graph has exactly two entity types. Study, registry, guideline,
and regulatory records are provenance attached to evidence relations, not
additional biomedical entity types.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClinicalUse(StrEnum):
    DIAGNOSTIC = "diagnostic"
    THERAPEUTIC = "therapeutic"


class RelationType(StrEnum):
    DIAGNOSTIC_HUMAN_EVIDENCE = "diagnostic_human_evidence"
    THERAPEUTIC_HUMAN_EVIDENCE = "therapeutic_human_evidence"
    THERANOSTIC_SELECTION_EVIDENCE = "theranostic_selection_evidence"


class CaseType(StrEnum):
    ATOMIC_SOURCE_RECORD = "atomic_source_record"
    PAIR_SNAPSHOT = "pair_snapshot"


class AtomicConclusion(StrEnum):
    SUPPORTIVE = "supportive"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"


class SnapshotConclusion(StrEnum):
    SUPPORTIVE = "supportive"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"
    CONFLICTING = "conflicting"


class DiagnosticMaturity(StrEnum):
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"


class TherapeuticMaturity(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


class PopulationScope(StrEnum):
    ADULT = "adult"
    PEDIATRIC = "pediatric"
    MIXED = "mixed"
    NOT_REPORTED = "not_reported"


class TumorSubtype(BaseModel):
    """Entity A: a formally standardized tumor subtype.

    Stage, target expression, age group, and prior treatment never live here;
    they belong to evidence applicability conditions.
    """

    model_config = ConfigDict(extra="forbid")

    tumor_subtype_id: str
    preferred_name: str
    canonical_system: str
    canonical_code: str
    aliases: list[str] = Field(default_factory=list)
    parent_tumor_subtype_id: str | None = None
    external_mappings: dict[str, list[str]] = Field(default_factory=dict)
    evidence_sources: list[str] = Field(default_factory=list)


class Radiopharmaceutical(BaseModel):
    """Entity B: radionuclide + ligand/carrier + identity-changing construct."""

    model_config = ConfigDict(extra="forbid")

    radiopharmaceutical_id: str
    preferred_name: str
    radionuclide: str
    ligand_or_carrier: str
    chemical_construct: str | None = None
    aliases: list[str] = Field(default_factory=list)
    external_mappings: dict[str, list[str]] = Field(default_factory=dict)
    evidence_sources: list[str] = Field(default_factory=list)


class ApplicabilityConditions(BaseModel):
    """Optional relation-level conditions copied only when the source reports them."""

    model_config = ConfigDict(extra="allow")

    population_scope: PopulationScope = PopulationScope.NOT_REPORTED
    disease_extent: str | None = None
    disease_status: str | None = None
    diagnostic_purpose: str | None = None
    target_selection_condition: str | None = None
    treatment_line: str | None = None
    other_clinical_conditions: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Traceability payload; source roles remain separate dimensions."""

    model_config = ConfigDict(extra="allow")

    source_id: str
    source_type: str
    record_locator: str
    evidence_public_date: date | None = None
    retrieved_at: datetime | None = None
    url: str | None = None
    doi: str | None = None
    pmid: str | None = None
    nct_id: str | None = None
    source_version: str | None = None
    license_or_access_status: str | None = None


class AtomicSourceRecordCase(BaseModel):
    """Reference answer for relation eligibility and atomic 3-class conclusion."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset_id: str
    case_type: Literal[CaseType.ATOMIC_SOURCE_RECORD]
    tumor_subtype_id: str | None
    radiopharmaceutical_id: str | None
    clinical_use: ClinicalUse
    cutoff_date: date
    provenance: Provenance
    reference_relation_eligible: bool
    reference_atomic_conclusion: AtomicConclusion | None
    study_family_id: str

    @model_validator(mode="after")
    def conclusion_matches_eligibility(self) -> AtomicSourceRecordCase:
        if self.reference_relation_eligible and self.reference_atomic_conclusion is None:
            raise ValueError("eligible atomic records require a three-class conclusion")
        if not self.reference_relation_eligible and self.reference_atomic_conclusion is not None:
            raise ValueError("ineligible candidates must use a null atomic conclusion")
        return self


class PairSnapshotCase(BaseModel):
    """Reference answer for pair-level maturity and four-class conclusion."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset_id: str
    case_type: Literal[CaseType.PAIR_SNAPSHOT]
    tumor_subtype_id: str
    radiopharmaceutical_id: str
    clinical_use: ClinicalUse
    cutoff_date: date
    provenance: Provenance
    reference_maturity_code: DiagnosticMaturity | TherapeuticMaturity
    reference_evidence_conclusion: SnapshotConclusion
    supporting_assertion_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def maturity_matches_use(self) -> PairSnapshotCase:
        code = self.reference_maturity_code.value
        if self.clinical_use == ClinicalUse.DIAGNOSTIC and not code.startswith("D"):
            raise ValueError("diagnostic snapshots require D1-D4")
        if self.clinical_use == ClinicalUse.THERAPEUTIC and not code.startswith("T"):
            raise ValueError("therapeutic snapshots require T1-T4")
        return self


ReferenceCase = Annotated[
    AtomicSourceRecordCase | PairSnapshotCase,
    Field(discriminator="case_type"),
]


class PredictionProvenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_id: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    generated_at: datetime | None = None


class AtomicPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset_id: str
    case_type: Literal[CaseType.ATOMIC_SOURCE_RECORD]
    clinical_use: ClinicalUse
    method_id: Literal["pure_llm", "naive_rag", "lightrag", "full_system"]
    run_id: int = Field(ge=1)
    predicted_tumor_subtype_id: str | None
    predicted_radiopharmaceutical_id: str | None
    predicted_relation_eligible: bool
    predicted_atomic_conclusion: AtomicConclusion | None
    prediction_provenance: PredictionProvenance | None = None

    @model_validator(mode="after")
    def conclusion_matches_prediction(self) -> AtomicPrediction:
        if self.predicted_relation_eligible and self.predicted_atomic_conclusion is None:
            raise ValueError("eligible predictions require an atomic conclusion")
        if not self.predicted_relation_eligible and self.predicted_atomic_conclusion is not None:
            raise ValueError("ineligible predictions require a null atomic conclusion")
        return self


class PairSnapshotPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    dataset_id: str
    case_type: Literal[CaseType.PAIR_SNAPSHOT]
    clinical_use: ClinicalUse
    method_id: Literal["pure_llm", "naive_rag", "lightrag", "full_system"]
    run_id: int = Field(ge=1)
    predicted_tumor_subtype_id: str | None
    predicted_radiopharmaceutical_id: str | None
    predicted_maturity_code: DiagnosticMaturity | TherapeuticMaturity
    predicted_evidence_conclusion: SnapshotConclusion
    prediction_provenance: PredictionProvenance | None = None

    @model_validator(mode="after")
    def maturity_matches_prediction_use(self) -> PairSnapshotPrediction:
        code = self.predicted_maturity_code.value
        if self.clinical_use == ClinicalUse.DIAGNOSTIC and not code.startswith("D"):
            raise ValueError("diagnostic predictions require D1-D4")
        if self.clinical_use == ClinicalUse.THERAPEUTIC and not code.startswith("T"):
            raise ValueError("therapeutic predictions require T1-T4")
        return self


BaselinePrediction = Annotated[
    AtomicPrediction | PairSnapshotPrediction,
    Field(discriminator="case_type"),
]


class EvidenceAssertion(BaseModel):
    """Graph edge payload; studies and status records remain provenance."""

    assertion_id: str
    tumor_subtype_id: str
    radiopharmaceutical_id: str
    relation_type: RelationType
    clinical_use: ClinicalUse
    atomic_conclusion: AtomicConclusion
    study_family_id: str
    applicability: ApplicabilityConditions = Field(default_factory=ApplicabilityConditions)
    provenance: Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)
