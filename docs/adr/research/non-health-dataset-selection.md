---
status: accepted
topic: research
created: 2026-11-25
updated: 2026-11-25
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

Each chosen or runner-up candidate's terms, and the fields and counts cited for the ten candidates
re-read here (FARS, CRSS, SBA 7(a) and 504, NIBRS, USAspending, TED, VCDB, NVD with KEV, NEISS and
GDELT), are quoted from the primary page, read on 2026-11-25. A quotation follows the selection
ADR's convention: the page's text with markup dropped and whitespace runs collapsed to one space,
running to a phrase end; a PDF's text is `pdftotext -layout` output; a JSON record is quoted as
served. Pages that blocked a plain fetch or render by script (NHTSA's FARS page, CPSC's NEISS
page, the NVD terms page, the FBI's Crime Data Explorer, USAspending) were read in a browser;
NHTSA's page stayed denied.
Every other candidate's terms are the research pass's reading, unchecked here.

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
   a coded field read against the rest. A threshold on a count or an amount is excluded, and so is
   an outcome that the candidate's own hierarchy determines.
2. **Two named cell axes**, each a field with one value per record, so the cells partition the
   records.
3. **An open machine-readable hierarchy** over a coded field of the record: a published file
   (XML, JSON, OWL, OBO or CSV) that states each code's parent. A code list inside a PDF manual or
   a statistical package's format catalogue is not one.
4. **Terms that permit the use**, quoted from the dataset's primary page, and for a hierarchy
   published apart from the dataset, from the hierarchy's page too.

The rule governs the non-health families only. For them it replaces condition 3 of the selection
rule ADR `research/confirmatory-dataset-selection` records; that rule's conditions 1, 2, 5 and 6
(open access, a per-record outcome, at least 5,000 records, publishable results) still hold, and
condition 4 there becomes condition 3 here.

**2. Verdicts.** A candidate is chosen when it meets all four conditions with its terms quoted
from the primary page. One whose terms are quoted but another condition is unquoted (a field, a
code list or a hierarchy's licence the pages read do not show) is a runner-up. A candidate whose
terms are unquoted is excluded under condition 4 whatever else it meets. HMDA and UCI
Census-Income KDD keep the verdicts ADR `research/confirmatory-dataset-selection` gives them,
cited in the table: KDD's census person record is the slot ACS PUMS holds.

**3. The ranking.** Best first: the chosen, then the runners-up, then the excluded. The excluded
come in two groups: first those whose terms page was read (here or by the selection ADR), then
those whose terms page was not read. Within a group they are ordered by the first condition each
fails, then alphabetically. For an unread candidate the failing condition is the research pass's
reading in the IDEAS entry.

| Rank | Candidate | Terms and fields, quoted from the primary page | Verdict |
|---|---|---|---|
| 1 | NVD CVE records with the CISA KEV label, over CWE ([`nvd-nd`](../../literature.md#datasets), [`kev-nd`](../../literature.md#datasets), [`cwe-nd`](../../literature.md#datasets)) | NVD: "The NVD API is intended to be used to develop a service or services to search, display, analyze, retrieve, view and otherwise "get" information from NVD data." KEV: "The KEV database is distributed under the Creative Commons 0 1.0 License." CWE: "The MITRE Corporation hereby grants you a non-exclusive, royalty-free license to use CWE for research, development, and commercial purposes." | Chosen, exploratory non-health family |
| 2 | VERIS Community Database ([`vcdb-nd`](../../literature.md#datasets)) | `LICENSE.txt`: "Creative Commons Attribution-ShareAlike 4.0 International Public License". Outcome, `attribute.confidentiality.data_disclosure`: "This is the core determiner if this incident is a breach." and "If this is 'Yes', it will be considered a breach." | Chosen, exploratory non-health family |
| 3 | EU TED CSV, 2006 to 2023, over CPV ([`ted-nd`](../../literature.md#datasets)) | Each distribution's licence label: "European Commission reuse notice" or "Creative Commons Attribution 4.0 International". Outcome: "1 = this notice was later cancelled" | Runner-up: CPV's licence is unquoted |
| 4 | SBA 7(a) and 504 FOIA loans ([`sba-nd`](../../literature.md#datasets)) | Licence: "http://www.usa.gov/publicdomain/label/1.0/"; "SBA 7(a) and 504 loan data reports for loans approved since FY1991." | Runner-up: the loan status codes are unread |
| 5 | UCI Census-Income KDD ([`censuskdd-2000`](../../literature.md#datasets)) | "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." | Runner-up, as ADR `research/confirmatory-dataset-selection` ranks it, below ACS PUMS |
| 6 | GDELT 2.0 Events ([`gdelt-nd`](../../literature.md#datasets)) | "all datasets released by the GDELT Project are available for unlimited and unrestricted use for any academic, commercial, or governmental use of any kind without fee." | Excluded, condition 1: the hierarchy determines the outcome |
| 7 | NIBRS | Licence of the FBI's record: "http://www.usa.gov/publicdomain/label/1.0/" | Excluded, condition 1: no per-incident outcome named |
| 8 | USAspending contract awards | "The data on this site is available to copy, adapt, redistribute, or otherwise use for non-commercial or for commercial purposes, subject to the Limitation on Permissible Use of Dun & Bradstreet, Inc. Data noted on the homepage." | Excluded, condition 1: single bid is a threshold on a count |
| 9 | FARS 2023 person file ([`fars-nd`](../../literature.md#datasets)) | `accessLevel` "public"; no licence field | Excluded, condition 3: codes in a PDF manual and SAS formats |
| 10 | NEISS 2024 | Licence of CPSC's record: "http://www.usa.gov/publicdomain/label/1.0/" | Excluded, condition 3: a flat product code list |
| 11 | HMDA ([`hmda-nd`](../../literature.md#datasets)) | No licence line on the page, as ADR `research/confirmatory-dataset-selection` records | Excluded, condition 4: no licence line |
| 12 | ATT&CK and CAPEC; EPSS; FIBO; GitHub Advisory Database; Lending Club and Kaggle sets; NTSB aviation, ASRS and FRA rail; O\*NET 31.0 and ESCO; ENVO, FoodOn, AGRO, PO, ChEBI and GO; OSV.dev; PatentsView with CPC; SEC EDGAR with US-GAAP XBRL; UCDP GED 26.1 | Terms not read | Excluded, condition 1: a knowledge source with no records, a score in place of a label, no outcome named, or (UCDP) a threshold on fatalities |
| 13 | CIC-IDS2017 and NSL-KDD; CRSS 2023; FDIC failed banks; NASA MDP and PROMISE; OULAD; Polish and Taiwanese bankruptcy sets; Stanford Open Policing; UCI Default of Credit Card Clients; UCI Dropout and Academic Success; UCI Student Performance; UNSW-NB15 | Terms not read; CRSS's page is on the host that denied FARS's | Excluded, condition 3: no hierarchy, free-text violations (Stanford), or (CRSS) FARS's codes |
| 14 | ACLED; Backblaze Drive Stats; CFPB Consumer Complaint Database; COMPAS two-year; Global Terrorism Database; ICEWS and Correlates of War; SIPRI arms transfers | Terms not read | Excluded, condition 4: terms not read, and on the research pass's reading ACLED, the GTD and SIPRI restrict the use and COMPAS states no licence |

Each re-read candidate against the four conditions, from its pages:

- **NVD with KEV.** A CVE record from the NVD API carries the KEV label inline: CVE-2021-44228's
  record holds `"cisaExploitAdd":"2021-12-10"` and a `weaknesses` list of CWE ids. On 2026-11-25
  the API reported 432,554 CVEs and 1,805 with `hasKev`. KEV membership is a two-valued outcome
  the record defines. The cells are CVSS `baseSeverity` by `attackVector`. CWE publishes its
  hierarchy as "XML.zip: A compressed XML file containing the desired CWE ID, dependent
  Weaknesses, Views, Categories, and all required External References." The NVD terms ask
  services to show "This product uses the NVD API but is not endorsed or certified by the NVD."
- **VCDB.** The outcome field takes `["No", "Potentially", "Yes", "Unknown"]`, and the schema
  itself names `Yes` the breach code, so the outcome is a named code. `victim.industry` is a NAICS
  code ("Victim NAICS Code.") and `victim.employee_count` one string from an enum, so the cells are
  NAICS sector by victim size. The actor field, the IDEAS entry's second axis, holds an object
  with keys `external`, `internal`, `partner` and `unknown`, so one incident can hold several actor
  types and it cannot partition the records. The hierarchy is the VERIS schema's action, variety
  and asset enumerations, a JSON file, with NAICS's prefix codes. The repository holds 11,097
  validated incident files. Its README says "Most VCDB issues are chosen randomly (with a
  preferences for those in the last year), however we specifically select healthcare issues and
  some priority incidents.", and some records are tagged "Encoded by AI".
- **TED.** The codebook defines `CANCELLED` with "1 = this notice was later cancelled", a
  two-valued outcome, and says "No information about cancellation is available for notices
  published before 2011." The cells are country by contract type. CPV is the hierarchy, and
  "There is a many-to-many matching between the old and the new CPV, so they cannot be aligned in
  a simple manner." CPV's EU Vocabularies page rendered no licence and no download list, its RDF
  record holds no licence, and the Publications Office's legal notice answered HTTP 429. The
  dataset record states no row count.
- **SBA 7(a) and 504.** The dataset page at data.sba.gov and the data dictionary both answered
  HTTP 404, so the loan status codes behind the charged-off outcome are unread. The data.gov
  record's licence is quoted above.
- **GDELT.** The codebook defines QuadClass with "This field specifies this primary
  classification for the event type, allowing analysis at the highest level of aggregation.", and
  "CAMEO event codes are defined in a three-level taxonomy." Material conflict is a function of
  the event code, so a frame would learn its label from its own hierarchy.
- **NIBRS.** The FBI's record says "It presents an overview of the offense categories that show
  the diversity and complexity of the data collected via NIBRS." The Crime Data Explorer offers
  "Incident-based data by state, summary data estimates, and data about other specific topics
  may be downloaded in CSV files from the selections below." No page read names a per-incident
  binary outcome.
- **USAspending.** The data dictionary's "Number of Offers Received" is a count; single bid is a
  threshold on it. The award-count endpoint reported `"contracts":5926883` for fiscal year 2024.
- **FARS.** "FARS data are made available to the public in Statistical Analysis System (SAS) data
  files as well as comma-separated values (CSV) files." The release notes say "Format files are
  provided as SAS Viya." The fatal outcome is `INJ_SEV` code 4, "Fatal Injury (K)", a named code;
  the codes and their groupings live in the analytical manual's tables, a PDF.
- **CRSS.** NHTSA's manuals page lists the shared "FARS/CRSS 2024 Coding and Validation Manual",
  so CRSS carries FARS's codes.
- **NEISS.** The outcome is a named code, "4 - Treated and Admitted/Hospitalized". CPSC's reports
  "summarize injury data by major product groups, product subgroups, age, and sex.", but the data
  dictionary lists `Product_1` and `Product_2` with no grouping, and no page read publishes the
  product grouping as a file.

**4. The chosen frames.** NVD with KEV over CWE, and VCDB. Each is its own exploratory family,
outside the health multiplicity family and apart from ACS PUMS. Each frame waits on the cell
machinery reading a frame's own two axes, since neither record describes a person. Each frame's
`Source` pin carries `agreement=None`, the value ADR `research/confirmatory-dataset-selection`
point 4 gives open data, so `permits_sending` lets the `bko` arm send its records. VCDB's
share-alike licence binds generated rows derived from it: any published fill carries the same
licence and VCDB's attribution.

## Rationale

The four conditions are what the machinery reads once the cell axes are a frame's own: a label
for the classifiers, two axes for the cell grid, and a hierarchy for gap detection. A named code
read against the rest is the dataset's own category, while a cutoff on a count is the frame's
choice, the reduction the selection rule's condition 2 refuses. GDELT's QuadClass is a named code
but a function of the event code, so a hierarchy-driven gap detector would be scored on a label
its own ontology defines. A hierarchy must be a file because the ontology loader reads files;
FARS's and NEISS's groupings sit in PDF tables and reports.

NVD with KEV ranks first: its terms are public-service, CC0 and royalty-free, its label sits
inline on every record, and CWE is a deep ChildOf hierarchy published as XML. VCDB ranks second:
its VERIS enumerations are shallow, its share-alike licence follows any fill, and its sample mixes
random and chosen incidents. Both clear the record minimum on the counts above.

TED meets every condition but one. Its outcome, axes and terms are quoted, and only CPV's own
licence is not. SBA's terms are quoted, but the status field that would hold its outcome was
unreachable. Both are runners-up. Each would be chosen by a re-read that quotes the missing item.

## Consequences

**Gained**: two non-health frames chosen from quoted terms, each with its outcome, axes and
hierarchy named before any record is loaded, and each candidate's exclusion stated.
**Accepted**: neither chosen hierarchy is OBO, so each frame unit converts CWE's XML or the VERIS
schema before the loader reads it. NVD holds over 400,000 records and VCDB about 11,000, so the
frame units fix their snapshot and any sampling before reading a gap. VCDB's non-random sample
and its AI-encoded records stay in the data; the frame unit records their shares. TED's
cancellation outcome exists only from 2011. Most candidates outside the ten re-read rest on the
research pass's reading of their terms.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| The person-axes condition for non-health frames | No vulnerability, incident, loan or tender record has a numeric age and a binary sex; the rule's condition 2 replaces it | 2026-11-25 |
| VCDB's cells as industry by actor type, the IDEAS entry's pair | An incident can hold several actor types, so actor type does not partition the records | 2026-11-25 |
| A threshold on a count as the outcome (USAspending single bid, UCDP fatalities) | The cutoff is the frame's choice, not the dataset's category | 2026-11-25 |
| GDELT with material conflict as the outcome | QuadClass is a function of the CAMEO event code, the hierarchy itself | 2026-11-25 |
| TED over CPV as a chosen frame | CPV's licence is unquoted: its EU Vocabularies page rendered none and the legal notice answered HTTP 429 | 2026-11-25 |
| SBA 7(a) and 504 as a chosen frame | The data dictionary and data.sba.gov's page answered HTTP 404, so the status codes are unread | 2026-11-25 |
| FARS and NEISS, the IDEAS entry's as-is picks | Each holds a coded outcome and person axes, but its hierarchy sits in PDF tables or reports, not a file | 2026-11-25 |

## Enforcement

- Owner: `docs/RESEARCH.md`
- Pinned by: nothing -- convention only

## Reversal Conditions

**A re-read of a quoted page differs from its quote**, in a licence, an outcome field or a
hierarchy's publication. **CPV's licence is quoted from its own page**, which makes TED a chosen
frame. **SBA's data dictionary is read and names a charged-off status code**, which makes SBA a
chosen frame. **A frame unit finds its hierarchy cannot be converted for the loader**, or its
record count below 5,000 after its snapshot, which excludes it and revises this record.
**The NVD API's terms or KEV's licence change.**

## Related

- [Confirmatory datasets are chosen by a fixed rule before any record is loaded, and each choice's terms are quoted from its primary page](confirmatory-dataset-selection.md)
- [ACS PUMS's frame: Massachusetts 2024, income above $50,000, occupation on the Occupation Ontology](acs-frame.md)
