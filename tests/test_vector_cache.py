"""Demo script for VectorStoreService.
Run: python tests/demo_vector_store_service.py

Requires OpenAI embeddings (OPENAI_API_KEY must be set).
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path BEFORE importing app.* modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.data_processor import get_pdf_processor
from app.services.vector_store_service import get_vector_store_service, VectorStoreService
from langchain_core.documents import Document

def _fallback_docs() -> list[Document]:
    samples = [
    "Clinical trial results indicate improved patient-reported outcomes in phase II.",
    "Adverse events were managed through dose titration and supportive therapy.",
    "Pharmacokinetic analysis shows rapid absorption and moderate half-life.",
    "Regulatory submission requires consolidated safety data and labeling strategy.",
    "Biomarker evaluations demonstrated strong correlation with therapeutic response.",
    "Efficacy endpoints were met with statistically significant improvement.",
    "Safety monitoring identified no unexpected treatment-emergent signals.",
    "Dose-escalation cohorts achieved predefined tolerability thresholds.",
    "Patient adherence rates remained high throughout the study timeline.",
    "Immunogenicity profiles were consistent with historical comparator data.",
    "Blinded review confirmed objective response rates across all arms.",
    "Quality-of-life metrics improved from baseline in most participants.",
    "The compound exhibited potent receptor binding activity in vitro.",
    "Data monitoring committee recommended continuation without modification.",
    "The investigational drug demonstrated durable clinical benefit.",
    "Interim analysis supports advancing to pivotal phase III evaluation.",
    "Real-world evidence suggests strong persistence and compliance trends.",
    "Exploratory endpoints revealed meaningful biomarker modulation.",
    "The therapy showed minimal drug–drug interaction risk.",
    "Safety database meets regulatory standards for global submission.",
    "Pharmacodynamic markers showed rapid onset of biological activity.",
    "The control arm maintained stable disease progression patterns.",
    "Post-marketing surveillance highlights favorable patient outcomes.",
    "The formulation achieved reliable bioavailability across populations.",
    "Protocol deviations were within acceptable operational limits.",
    "Study retention exceeded projections for all trial sites.",
    "Patient stratification improved signal detection across subsets.",
    "The molecule demonstrated high selectivity for its target pathway.",
    "Clinical investigators reported consistent tolerability across visits.",
    "The adverse event profile aligned with the known class effects.",
    "Final readout confirmed robust efficacy across disease severity levels.",
    "Dosing recommendations were refined based on exposure–response modeling.",
    "The regulatory package includes comprehensive safety narratives.",
    "Secondary endpoints showed incremental clinical improvements.",
    "Patient diaries indicated reduced symptom burden after treatment.",
    "The screening period identified key eligibility challenges.",
    "Pharmacokinetic variability was low across demographic groups.",
    "The treatment arm displayed higher remission rates.",
    "The safety committee approved expansion into additional cohorts.",
    # "Antibody titers remained stable with repeated dosing.",
    # "In vivo studies demonstrated predictable therapeutic effects.",
    # "Cross-functional teams aligned on submission timelines.",
    # "Risk-management plans were updated per regulatory guidance.",
    # "Site feasibility assessments highlighted strong enrollment potential.",
    # "The assay demonstrated reproducibility across validation runs.",
    # "Investigators observed early signs of clinical stabilization.",
    # "Label expansion opportunities are supported by emerging evidence.",
    # "Statistical modeling confirmed robustness of the primary endpoint.",
    # "Comparative analysis showed improved tolerability over standard care.",
    # "Drug exposure remained within the predefined therapeutic window.",
    # "Quality checks ensured data integrity throughout the trial.",
    # "Sensitivity analyses validated the strength of efficacy signals.",
    # "Operational metrics indicated efficient trial-site performance.",
    # "Enrichment strategies improved recruitment of target populations.",
    # "Plasma concentration profiles supported once-daily dosing.",
    # "No clinically meaningful changes in laboratory parameters were observed.",
    # "Patient education materials were optimized for comprehension.",
    # "Dose-response curves demonstrated clear saturation at higher ranges.",
    # "Harmonized global protocols facilitated streamlined execution.",
    # "Long-term extension studies indicate sustained clinical benefit.",
    # "The compound exhibited low clearance and moderate distribution volume.",
    # "Clinical supplies were delivered on schedule across regions.",
    # "Evidence synthesis confirmed alignment with real-world outcomes.",
    # "Records audit showed strong compliance with GCP requirements.",
    # "Mechanistic studies revealed unique pathway inhibition properties.",
    # "Study randomization achieved balanced demographic distribution.",
    # "Adherence monitoring tools reduced protocol deviations.",
    # "Risk-benefit assessment remains favorable across all populations.",
    # "Preliminary readout shows rapid symptomatic improvement.",
    # "Multi-omics analysis supports the proposed mechanism of action.",
    # "Enrollment pace accelerated after protocol amendment.",
    # "High assay sensitivity enabled early biomarker detection.",
    # "Investigators reported improved patient mobility and energy levels.",
    # "Statistical review ensured valid interpretation of subgroup effects.",
    # "Real-time data monitoring enhanced trial oversight.",
    # "Model-based simulations informed optimal dosing regimens.",
    # "Recruitment strategies improved diversity across trial sites.",
    # "Data reconciliation efforts ensured accurate safety reporting.",
    # "In vitro potency exceeded benchmarks for first-in-class candidates.",
    # "The drug demonstrated favorable tolerability in elderly populations.",
    # "The therapeutic index remained wide across dosing schedules.",
    # "Site audits confirmed compliance with regulatory expectations.",
    # "Updated guidance prompted enhancements to labeling content.",
    # "Follow-up assessments documented durable response trajectories.",
    # "Comparative safety review identified no new areas of concern.",
    # "The clinical program benefits from strong investigator engagement.",
    # "Biomarker-driven segmentation revealed distinct responder profiles.",
    # "Longitudinal data supported consistent improvements over time.",
    # "Review board approved the revised informed-consent materials.",
    # "Analysis of covariance confirmed treatment-related differences.",
    # "Operational dashboards improved visibility into study progress.",
    # "Early-phase studies demonstrated promising biological activity.",
    # "Baseline characteristics were comparable between study arms.",
    # "Drug metabolism followed expected enzymatic pathways.",
    # "Patient satisfaction scores increased post-treatment.",
    # "Trial governance processes ensured adequate oversight.",
    # "Adaptive design features enabled efficient resource utilization.",
    # "Cardiac monitoring indicated stable ECG parameters.",
    # "The investigational therapy maintained efficacy across weight ranges.",
    "Cross-study comparisons reinforced confidence in the clinical profile."]

    return [Document(page_content=s, metadata={"source": "sample", "line": i}) for i, s in enumerate(samples)]

def main():
    total_start = time.perf_counter()

    docs = _fallback_docs()
    try:
        vs_service = get_vector_store_service(embedding_provider="gemini", auto_load=True, use_cache=True)
    except EnvironmentError as e:
        print(f"[WARN] {e}. Falling back to sample in-memory docs with mock build.")

    print(f"Vector store initialized with {vs_service.count()} documents using provider='{vs_service.embedding_provider}'.")

    vs_service.add_documents(docs)

    total_end = time.perf_counter()
    total_elapsed = total_end - total_start
    print("\n=== Timing Summary ===")
    print(f"Total runtime: {total_elapsed:.3f}s")

if __name__ == "__main__":
    main()
