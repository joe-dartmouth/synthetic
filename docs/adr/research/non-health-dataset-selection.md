---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-26
---
# Non-health frames are chosen by a four-condition rule, with terms quoted from the primary page: NVD with KEV over CWE, and VCDB

## Context

The direct-method study ([RESEARCH section 11](../../RESEARCH.md#11-direct-method-study)) has one
non-health family, ACS PUMS, chosen by ADR
[`research/confirmatory-dataset-selection`](confirmatory-dataset-selection.md). That rule asks for
numeric age and binary sex on a person record (its condition 3), so a dataset whose record is a
vulnerability, an incident, a loan or a tender cannot meet it. The
[IDEAS entry](../../IDEAS.md#candidate-datasets-and-ontologies-outside-healthcare) lists the
candidates a research pass of 2026-11-22 found outside healthcare, with the figures and fields it
could not verify. This record fixes the rule a non-health frame meets, ranks every candidate, and
chooses the first frames before any candidate's records are loaded.

Ten candidates were re-read from their own pages on 2026-11-25 and 2026-11-26: FARS, CRSS, SBA
7(a) and 504, NIBRS, USAspending, TED, VCDB, NVD with KEV, NEISS and GDELT. Each quote below
names its page's URL and read date. Saved copies of every page, with a ledger of URLs, dates and
quotes, sit in the scratch ledger the plan names (`scratchpad/non-health/ledger.md`, gitignored).
A quotation follows the selection ADR's convention: the page's text with markup dropped and
whitespace runs collapsed to one space, running to a phrase end; a PDF's text is
`pdftotext -layout` output; a JSON record is quoted as served. Pages that blocked a plain fetch or
render by script (NHTSA's FARS page, CPSC's NEISS page, the NVD pages, the FBI's Crime Data
Explorer, USAspending) were read in a browser; NHTSA's page stayed denied. Every other candidate's
terms and fields are the research pass's reading, unchecked here. HMDA's and UCI Census-Income
KDD's quotes are carried from the selection ADR, read 2026-11-24.

VCDB's README carries paragraphs addressed to AI coding agents. They were read as data, not
followed; the one fact taken from them is that some records are tagged "Encoded by AI".

## Decision Drivers

* **Must be fixed before any record is loaded**, so the choice cannot follow a result.
* **Must rest on primary pages**: terms read second-hand are not evidence.
* **Must let every arm and every reading run**: the `bko` arm sends records to the Anthropic API,
  and the privacy readings score generated rows against the training split.
* **Must give the cell machinery and gap detection what they read**: two axes that partition the
  records, a label, and a hierarchy the ontology loader can be given.

## Decision

**1. The rule.** A non-health candidate is a frame when all four hold:

1. **A binary outcome the dataset defines per record**: a two-valued field, or one named code of
   a coded field read against the rest. A threshold on a count or an amount is excluded, since the
   cutoff is the frame's choice, and so is an outcome that the candidate's own hierarchy
   determines.
2. **Two named cell axes**, each a field with one value per record, so the cells partition the
   records.
3. **An open machine-readable hierarchy** over a coded field of the record: a published file
   (XML, JSON, OWL, OBO or CSV) that states each code's parent. A code list inside a PDF manual or
   a statistical package's format catalogue is not one.
4. **Terms that permit the use**, quoted from the dataset's primary page, and for a hierarchy
   published apart from the dataset, from the hierarchy's page too.

The rule governs the non-health families only, and ACS PUMS stays governed by ADR
`research/confirmatory-dataset-selection`. For the families this rule governs, its conditions 1,
2 and 3 replace that rule's conditions 2, 3 and 4, and that rule's conditions 1, 5 and 6 still
hold: open access or an agreement the team holds, at least 5,000 records, and terms and
documentation that let results be published.

**2. Verdicts.** A candidate is chosen when it meets all four conditions, each quoted from a page
read. A candidate whose own data terms are quoted, but with another item unquoted (a field, an
axis, the hierarchy's file or the hierarchy's licence), is a runner-up. A candidate whose own data
terms are unquoted is excluded. HMDA and UCI Census-Income KDD are ranked again under this rule,
with their quotes carried from the selection ADR: KDD's income label is a threshold on an amount
(condition 1 here), and HMDA, which the selection ADR fails on its person-axes condition, has no
licence line (condition 4 here).

**3. The ranking.** Best first: the chosen, then the runners-up, then the excluded. The chosen are
ordered as Rationale gives, the runners-up alphabetically. The excluded come in three groups:
those whose terms page was read, here or by the selection ADR; those whose terms page was not
read; and those the research pass did not check. Within the first two groups they are ordered by
the first condition each fails, then alphabetically. The row label names that condition; where an
earlier condition was not read, it names the first condition read to fail. Unquoted terms set the
group, not the label. For an unread candidate the failing condition is the research pass's reading
in the IDEAS entry.

| Rank | Candidate | Terms and fields, quoted, with page and read date | Verdict |
|---|---|---|---|
| 1 | NVD CVE records with the CISA KEV label, over CWE ([`nvd-nd`](../../literature.md#datasets), [`kev-nd`](../../literature.md#datasets), [`cwe-nd`](../../literature.md#datasets)) | NVD (nvd.nist.gov/general/FAQ-Sections/General-FAQs, 2026-11-26), answering "How can my organization use the NVD data within our own products and services?": "All NIST publications are available in the public domain." NVD API (nvd.nist.gov/developers/terms-of-use, 2026-11-25): "The NVD API is intended to be used to develop a service or services to search, display, analyze, retrieve, view and otherwise "get" information from NVD data." KEV (github.com/cisagov/kev-data `LICENSE`, 2026-11-25): "The KEV database is distributed under the Creative Commons 0 1.0 License." CWE (cwe.mitre.org/about/termsofuse.html, 2026-11-25): "The MITRE Corporation hereby grants you a non-exclusive, royalty-free license to use CWE for research, development, and commercial purposes." | Chosen, exploratory non-health family |
| 2 | VERIS Community Database, over the VERIS schema ([`vcdb-nd`](../../literature.md#datasets), [`veris-nd`](../../literature.md#datasets)) | VCDB (github.com/vz-risk/VCDB `LICENSE.txt`, 2026-11-25) and VERIS (github.com/vz-risk/veris `LICENSE.txt`, 2026-11-26), each: "Creative Commons Attribution-ShareAlike 4.0 International Public License". Outcome, `attribute.confidentiality.data_disclosure` (VCDB `vcdb-merged.json`, 2026-11-25): "This is the core determiner if this incident is a breach." and "If this is 'Yes', it will be considered a breach." | Chosen, exploratory non-health family |
| 3 | SBA 7(a) and 504 FOIA loans ([`sba-nd`](../../literature.md#datasets)) | Licence of the data.gov record (catalog.data.gov/dataset/sba-7a-and-504-loan-data-reports, 2026-11-25): "http://www.usa.gov/publicdomain/label/1.0/"; "SBA 7(a) and 504 loan data reports for loans approved since FY1991." | Runner-up: the loan status codes, the state by term band axes and the NAICS file are unquoted |
| 4 | EU TED CSV, 2006 to 2023, over CPV ([`ted-nd`](../../literature.md#datasets)) | Licence label of each distribution (data.europa.eu dataset `ted-csv`, 2026-11-25): "European Commission reuse notice" or "Creative Commons Attribution 4.0 International". Outcome (the TED CSV Notes and Codebook v3.6.1, 2026-11-25): "1 = this notice was later cancelled" | Runner-up: CPV's licence, CPV's download list and the country by contract type axes are unquoted |
| 5 | GDELT 2.0 Events ([`gdelt-nd`](../../literature.md#datasets)) | gdeltproject.org/about.html, 2026-11-25: "all datasets released by the GDELT Project are available for unlimited and unrestricted use for any academic, commercial, or governmental use of any kind without fee." | Excluded, condition 1: the hierarchy determines the outcome |
| 6 | NIBRS ([`nibrs-nd`](../../literature.md#datasets)) | Licence of the FBI's data.gov record (catalog.data.gov harvest record e2908629, 2026-11-25): "http://www.usa.gov/publicdomain/label/1.0/" | Excluded, condition 1: no per-incident outcome named |
| 7 | UCI Census-Income KDD ([`censuskdd-2000`](../../literature.md#datasets)) | Carried from ADR `research/confirmatory-dataset-selection` (archive.ics.uci.edu/dataset/117, 2026-11-24): "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." | Excluded, condition 1: income over $50,000 is a threshold on an amount |
| 8 | USAspending contract awards ([`usaspending-nd`](../../literature.md#datasets)) | usaspending.gov/about, 2026-11-25: "The data on this site is available to copy, adapt, redistribute, or otherwise use for non-commercial or for commercial purposes, subject to the Limitation on Permissible Use of Dun & Bradstreet, Inc. Data noted on the homepage." | Excluded, condition 1: single bid is a threshold on a count |
| 9 | FARS 2023 person file ([`fars-nd`](../../literature.md#datasets)) | data.gov record (catalog.data.gov/dataset/fatality-analysis-reporting-system-fars, 2026-11-25): `accessLevel` "public"; no licence field | Excluded, condition 3: codes in a PDF manual and SAS formats |
| 10 | NEISS 2024 ([`neiss-nd`](../../literature.md#datasets)) | Licence of CPSC's data.gov record (catalog.data.gov harvest record be22731f, 2026-11-25): "http://www.usa.gov/publicdomain/label/1.0/" | Excluded, condition 3: a flat product code list |
| 11 | HMDA ([`hmda-nd`](../../literature.md#datasets)) | Carried from ADR `research/confirmatory-dataset-selection` (ffiec.cfpb.gov/data-publication, 2026-11-24): no licence line on the page | Excluded, condition 4: no licence line; conditions 1 to 3 not read under this rule |
| 12 | ATT&CK and CAPEC | Terms not read | Excluded, condition 1: a knowledge source with no records |
| 13 | ENVO, FoodOn, AGRO, PO, ChEBI and GO | Terms not read | Excluded, condition 1: no tabular dataset with a binary outcome found |
| 14 | EPSS | Terms not read | Excluded, condition 1: a probability score, not a label |
| 15 | FIBO | Terms not read | Excluded, condition 1: a knowledge source with no records |
| 16 | GitHub Advisory Database | Terms not read | Excluded, condition 1: no outcome named |
| 17 | O\*NET 31.0 and ESCO | Terms not read | Excluded, condition 1: knowledge sources with no records |
| 18 | UCDP GED 26.1 | Terms not read | Excluded, condition 1: fatalities over a threshold |
| 19 | CIC-IDS2017 and NSL-KDD | Terms not read | Excluded, condition 3: no hierarchy |
| 20 | COMPAS two-year | Terms not read; the research pass read no licence | Excluded, condition 3: no hierarchy |
| 21 | CRSS 2023 ([`crss-nd`](../../literature.md#datasets)) | Terms not read; its manuals list (static.nhtsa.gov CRSS manuals PDF, 2026-11-25) names the shared "FARS/CRSS 2024 Coding and Validation Manual" | Excluded, condition 3: FARS's codes |
| 22 | FDIC failed banks | Terms not read | Excluded, condition 3: a weak hierarchy |
| 23 | NASA MDP and PROMISE | Terms not read | Excluded, condition 3: a weak hierarchy |
| 24 | OULAD | Terms not read | Excluded, condition 3: a weak hierarchy |
| 25 | Polish and Taiwanese bankruptcy sets | Terms not read | Excluded, condition 3: a weak hierarchy |
| 26 | Stanford Open Policing | Terms not read | Excluded, condition 3: violation text without a code system |
| 27 | UCI Default of Credit Card Clients | Terms not read | Excluded, condition 3: no hierarchy |
| 28 | UCI Dropout and Academic Success | Terms not read | Excluded, condition 3: no hierarchy |
| 29 | UCI Student Performance | Terms not read | Excluded, condition 3: no hierarchy |
| 30 | UNSW-NB15 | Terms not read; the research pass read academic research only | Excluded, condition 3: no hierarchy |
| 31 | ACLED | Terms not read; the research pass read a bar on training use without written consent | Excluded, condition 4 |
| 32 | Backblaze Drive Stats | Terms not read | Excluded, condition 4 |
| 33 | CFPB Consumer Complaint Database | Terms not read | Excluded, condition 4 |
| 34 | Global Terrorism Database | Terms not read; the research pass read a bar on redistribution | Excluded, condition 4 |
| 35 | ICEWS and Correlates of War | Terms not read; the research pass read terms that weaken them | Excluded, condition 4 |
| 36 | SIPRI arms transfers | Terms not read; the research pass read a licence needed for commercial use | Excluded, condition 4 |
| 37 | Lending Club and Kaggle sets; NTSB aviation, ASRS and FRA rail; OSV.dev; PatentsView with CPC; SEC EDGAR with US-GAAP XBRL | Not checked by the research pass | Excluded: not checked |

Each re-read candidate against the four conditions, from its pages:

- **NVD with KEV.** The CVE API (services.nvd.nist.gov/rest/json/cves/2.0, read 2026-11-25 local,
  2026-11-26 UTC) returns CVE-2021-44228 with `"cisaExploitAdd":"2021-12-10"` and a `weaknesses`
  list of CWE ids. `cisaExploitAdd` appears only on a KEV record, so its absence reads as not in
  KEV: a two-valued outcome the record defines. The same endpoint reported 432,554 CVEs
  (`resultsPerPage=1`) and 1,805 with `hasKev`. The record can carry several CVSS metric sets
  (CVE-2021-44228 holds a `Primary` and a `Secondary` `cvssMetricV31` and a `cvssMetricV2`); the
  cells are `baseSeverity` by `attackVector` of the `cvssMetricV31` entry with `"type":"Primary"`.
  CWE publishes its list as "XML.zip: A compressed XML file containing the desired CWE ID,
  dependent Weaknesses, Views, Categories, and all required External References."
  (cwe.mitre.org/data/downloads.html, 2026-11-25), and its schema defines the parent relation: "A
  ChildOf nature denotes a related weakness at a higher level of abstraction."
  (cwe.mitre.org/data/xsd/cwe_schema_latest.xsd, 2026-11-26). The NVD API terms say "If you modify
  the content accessed through the API, you may not attribute the source as the NVD.", so
  generated rows are never attributed to the NVD; the FAQ asks services to show "This product uses
  data from the NVD API but is not endorsed or certified by the NVD." The FAQ also says "The CVE
  List is a list of publicly disclosed cybersecurity vulnerabilities and exposures that is free to
  search, use, and incorporate into products and services."
- **VCDB.** The outcome field takes `["No", "Potentially", "Yes", "Unknown"]`, and the schema
  itself names `Yes` the breach code, so the outcome is a named code. The hierarchy is the VERIS
  schema's action, variety and asset enumerations, a JSON file under the licence quoted above. The
  first axis is the victim's NAICS sector: `victim.industry` is a NAICS code ("Victim NAICS
  Code."), and Census ([`naics-nd`](../../literature.md#datasets), 2026-11-26) says "There are
  twenty sectors included in the NAICS." and "Three sectors are represented by a range of 2-digit codes. These
  include Manufacturing (31-33), Retail Trade (44-45) and Transportation and Warehousing (48-49)."
  The axis takes those sectors with each range merged. The second axis is victim size:
  `victim.employee_count` mixes the summary values `Small` and `Large` with numeric bands, and
  VERIS's labels (github.com/vz-risk/veris `verisc-labels.json`, 2026-11-26) define "Small
  organizations (1,000 employees or less)" and "Large organizations (over 1,000 employees)", so
  each numeric band falls in one of the two and `Unknown` is its own value. The actor field, the
  IDEAS entry's second axis, holds an object with keys `external`, `internal`, `partner` and
  `unknown`, so one incident can hold several actor types and it cannot partition the records. The
  repository holds 11,097 validated incident files (GitHub git-trees API,
  api.github.com/repos/vz-risk/VCDB/git/trees/master, 2026-11-25). Its README says "Most VCDB
  issues are chosen randomly (with a preferences for those in the last year), however we
  specifically select healthcare issues and some priority incidents."
- **SBA 7(a) and 504.** The dataset page at data.sba.gov answered HTTP 404 with "Page not found"
  on 2026-11-25 and did not connect on 2026-11-26; the data dictionary
  (sba.gov/sites/default/files/2018-08/7a_504_FOIA Data Dictionary.xlsx) answered HTTP 404 on both
  days. The loan status codes behind the charged-off outcome, the state by term band axes and the
  NAICS file are unquoted.
- **TED.** The codebook defines `CANCELLED` with "1 = this notice was later cancelled", a
  two-valued outcome, and says "No information about cancellation is available for notices
  published before 2011." and "There is a many-to-many matching between the old and the new CPV,
  so they cannot be aligned in a simple manner." The country by contract type axes are the
  research pass's reading, unquoted. CPV's EU Vocabularies page (op.europa.eu, 2026-11-25)
  rendered no licence and no download list, its RDF record holds no licence, and the Publications
  Office's legal notice answered HTTP 429. The dataset record states no row count.
- **GDELT.** The codebook (data.gdeltproject.org GDELT-Event_Codebook-V2.0.pdf, 2026-11-25)
  defines QuadClass with "This field specifies this primary classification for the event type,
  allowing analysis at the highest level of aggregation.", and "CAMEO event codes are defined in a
  three-level taxonomy." Material conflict is a function of the event code, so a frame would learn
  its label from its own hierarchy.
- **NIBRS.** The FBI's data.gov record says "It presents an overview of the offense categories
  that show the diversity and complexity of the data collected via NIBRS." The Crime Data Explorer
  (cde.ucr.cjis.gov downloads page, 2026-11-25) offers "Incident-based data by state, summary data
  estimates, and data about other specific topics may be downloaded in CSV files from the
  selections below." No page read names a per-incident binary outcome.
- **USAspending.** The data dictionary (usaspending.gov/data-dictionary, 2026-11-25) lists "Number
  of Offers Received", a count; single bid is a threshold on it. The award-count endpoint
  (api.usaspending.gov/api/v2/search/spending_by_award_count/, fiscal year 2024, 2026-11-25)
  reported `"contracts":5926883`.
- **FARS.** The analytical user's manual, 1975-2024 (crashstats.nhtsa.dot.gov publication 813794,
  2026-11-25) says "FARS data are made available to the public in Statistical Analysis System
  (SAS) data files as well as comma-separated values (CSV) files." The 2023 release notes say
  "Format files are provided as SAS Viya." The fatal outcome is `INJ_SEV` code 4, "Fatal Injury
  (K)", a named code; the codes and their groupings live in the manual's tables, a PDF.
- **CRSS.** NHTSA's CRSS manuals list names the shared "FARS/CRSS 2024 Coding and Validation
  Manual", so CRSS carries FARS's codes.
- **NEISS.** The outcome is a named code, "4 - Treated and Admitted/Hospitalized" (CPSC data
  dictionary DataDictionary042022.xlsx, 2026-11-25). CPSC's NEISS page (2026-11-25) says its
  reports "summarize injury data by major product groups, product subgroups, age, and sex.", but
  the data dictionary lists `Product_1` and `Product_2` with no grouping, and no page read
  publishes the product grouping as a file.

**4. The chosen frames.** NVD with KEV over CWE, and VCDB over the VERIS schema. Each is its own
exploratory family, outside the health multiplicity family and apart from ACS PUMS. Each frame
waits on the cell machinery reading a frame's own two axes, since neither record describes a
person. Each frame's `Source` pin carries `agreement=None`, the value ADR
`research/confirmatory-dataset-selection` point 4 gives open data, so `permits_sending` lets the
`bko` arm send its records. VCDB's and VERIS's licence grants the right to "produce, reproduce, and
Share Adapted Material." No page read states whether a generated row is Adapted Material, so the
share-alike condition is taken to bind, the conservative reading the project owner chose on
2026-11-26: any published fill derived from VCDB carries CC BY-SA 4.0 and VCDB's attribution.

## Rationale

The four conditions are what the machinery reads once the cell axes are a frame's own: a label
for the classifiers, two axes for the cell grid, and a hierarchy for gap detection. A named code
read against the rest is the dataset's own category, while a cutoff on a count or an amount is the
frame's choice, the reduction the selection rule's condition 2 refuses. GDELT's QuadClass is a
named code but a function of the event code, so a hierarchy-driven gap detector would be scored on
a label its own ontology defines. A hierarchy must be a file because the ontology loader reads
files; FARS's and NEISS's groupings sit in PDF tables and reports.

NVD with KEV ranks first: its terms are public domain, CC0 and royalty-free, its label is read
from every record, and CWE's ChildOf relation is published in XML. VCDB ranks second: its VERIS
enumerations are shallower, its published fills carry the licence obligation of Decision 4, and its
sample mixes random and chosen incidents. Both clear the record minimum on the counts above.

SBA and TED have their data terms quoted but leave other items unquoted, so each is a runner-up.
Each would be chosen by a re-read that quotes what it lacks.

## Consequences

**Gained**: two non-health frames chosen from quoted terms, each with its outcome, axes and
hierarchy named before any record is loaded, and each candidate's exclusion stated.
**Accepted**: neither chosen hierarchy is OBO, so each frame unit converts CWE's XML or the VERIS
schema before the loader reads it. The KEV label is rare: 1,805 of 432,554 CVEs, about 0.42%, on
the API counts above, so the NVD frame unit fixes its snapshot and any sampling before reading a
gap. VCDB's non-random sample and its AI-encoded records stay in the data; the frame unit records
their shares. VCDB's published fills carry the licence obligation of Decision 4. TED's cancellation
outcome exists only from 2011. Most candidates outside the ten re-read rest on the research pass's
reading of their terms.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The person-axes condition for non-health frames | No vulnerability, incident, loan or tender record has a numeric age and a binary sex; this rule's condition 2 replaces it | 2026-11-25 |
| VCDB's cells as industry by actor type, the IDEAS entry's pair | An incident can hold several actor types, so actor type does not partition the records | 2026-11-25 |
| Publishing a VCDB-derived fill without CC BY-SA 4.0 and VCDB's attribution | No page read says a generated row is not Adapted Material, so such a fill could breach the licence (Decision 4) | 2026-11-26 |
| NAICS as part of VCDB's hierarchy | The VERIS enumerations are the hierarchy the frame reads; NAICS sectors serve only as a cell axis | 2026-11-26 |
| A threshold on a count or an amount as the outcome (USAspending single bid, UCDP fatalities, KDD income) | The cutoff is the frame's choice, not the dataset's category | 2026-11-26 |
| GDELT with material conflict as the outcome | QuadClass is a function of the CAMEO event code, the hierarchy itself | 2026-11-25 |
| TED over CPV as a chosen frame | CPV's licence and download list are unquoted, and so are its axes | 2026-11-26 |
| SBA 7(a) and 504 as a chosen frame | The status codes, axes and NAICS file are unquoted; data.sba.gov and the dictionary did not serve | 2026-11-26 |
| FARS and NEISS, the IDEAS entry's as-is picks | Each holds a coded outcome and person axes, but its hierarchy sits in PDF tables or reports, not a file | 2026-11-25 |

## Enforcement

- Owner: `docs/RESEARCH.md`
- Pinned by: nothing -- convention only

## Reversal Conditions

**A re-read of a quoted page differs from its quote**, in a licence, an outcome field or a
hierarchy's publication. **CPV's licence and download list and TED's axes are quoted from their
pages**, which makes TED a chosen frame. **SBA's data dictionary is read and names a charged-off
status code, with its axes and NAICS file quoted**, which makes SBA a chosen frame. **A page
states that a generated row is not Adapted Material under VCDB's licence**, which lifts the
share-alike condition from its fills. **A frame unit finds its hierarchy cannot be converted for
the loader**, or its record count below 5,000 after its snapshot, which excludes it and revises
this record. **The NVD terms, the NVD FAQ's public-domain
statement or KEV's licence change.**

## Related

- [Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page](confirmatory-dataset-selection.md)
- [ACS PUMS's frame: Massachusetts 2024, income above $50,000, occupation on the Occupation Ontology](acs-frame.md)
