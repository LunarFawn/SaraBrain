# Ch10 Teaching Gaps

Auto-generated from `run_spacy_ch10.py --report-gaps`. Each entry is a question Sara got wrong or abstained on, classified by the kind of teaching it needs.

## Summary

| Kind | Count | Teach priority |
|---|---|---|
| relation_gap | 6 | 1 |
| distinction_gap | 2 | 2 |
| vocab_gap | 18 | 3 |

Priority ordering: relation > distinction > vocab. Relation-gap fixes reuse vocabulary Sara already has; vocab-gap fixes require teaching new terms from scratch and are cheaper when paired with a fact that uses them.

## relation_gap  (6)

*Sara knows all the words, but no fact in her graph connects the question's subject to the correct answer. Teach the connecting fact.*

### Q54  (abstain)

- **Question:** During which phase of the cell cycle does the quantity of DNA in a eukaryotic cell typically double?
- **Correct answer:** S
- **Unknown in question:** `quantity`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q84  (wrong)

- **Question:** All of the following are modes of asexual reproduction EXCEPT
- **Correct answer:** meiosis
- **Unknown in question:** `except`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q95  (abstain)

- **Question:** Imagine an organism whose 2n = 96. Meiosis would leave this organism’s cells with how many chromosomes?
- **Correct answer:** 48
- **Unknown in question:** `imagine, leave`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q184  (tie)

- **Question:** In a hypothetical population's gene pool, an autosomal gene, which had previously been fixed, undergoes a mutation that introduces a new allele, one inherited according to incomplete dominance. Natural selection then causes stabilizing selection at this locus. Consequently, what should happen over the course of many generations?
- **Correct answer:** Both A and C
- **Unknown in question:** `hypothetical, population, pool, autosomal, fix, undergo, introduce, inherit, accord, incomplete, dominance, natural, cause, locus, happen, course, generation`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q227  (abstain)

- **Question:** snRNPs are most closely associated with which of the following?
- **Correct answer:** RNA processing
- **Unknown in question:** `snrnps, associate`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q278  (abstain)

- **Question:** Crossing-over occurs during which of the following phases in meiosis?
- **Correct answer:** Prophase I
- **Unknown in question:** `follow`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

## distinction_gap  (2)

*A linking fact exists, but it doesn't discriminate between the MC choices. Teach a finer-grained fact that distinguishes them.*

### Q191  (tie)

- **Question:** The geneticist Mary Lyon hypothesized the existence of structures visible just under the nuclear membrane in mammals, which were later named Barr bodies. Which of the following statement is NOT correct about Barr bodies?
- **Correct answer:** The same chromosome in every cell of a normal female is inactivated.
- **Unknown in question:** `geneticist, mary, lyon, hypothesize, existence, visible, membrane, mammal, name, barr, body, follow, statement`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q216  (abstain)

- **Question:** Which of the following disorders will most likely result from X chromosome aneuploidy in women?
- **Correct answer:** Turner syndrome
- **Unknown in question:** `follow, disorder, aneuploidy, woman`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

## vocab_gap  (18)

*Sara has never seen one or more key terms in the correct answer. Teach a definition for each unknown term.*

### Q5  (abstain)

- **Question:** Which of the following would most likely provide examples of mitotic cell divisions?
- **Correct answer:** longitudinal section of a shoot tip
- **Unknown in question:** `provide`
- **Unknown in answer:** `longitudinal, section, shoot, tip`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q22  (wrong)

- **Question:** Mules are relatively long-lived and hardy organisms that cannot, generally speaking, perform successful meiosis. Consequently, which statement about mules is true?
- **Correct answer:** They have a relative evolutionary fitness of zero.
- **Unknown in question:** `live, hardy, speak, successful, statement`
- **Unknown in answer:** `relative, zero`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q25  (abstain)

- **Question:** Which of the following statements best describes what a Barr body is and its significance?
- **Correct answer:** It is an inactivated X chromosome and results in females with half their cells having one X inactivated and the other half of their cells having the other X inactivated.
- **Unknown in question:** `follow, statement, good, describe, barr, body, significance`
- **Unknown in answer:** `inactivated`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q31  (abstain)

- **Question:** The development of an egg without fertilization is known as
- **Correct answer:** parthenogenesis
- **Unknown in question:** `know`
- **Unknown in answer:** `parthenogenesis`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q48  (abstain)

- **Question:** In humans, fertilization normally occurs in the
- **Correct answer:** fallopian tube
- **Unknown in answer:** `fallopian, tube`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q61  (tie)

- **Question:** Crossover would most likely occur in which situation?
- **Correct answer:** Gene 1 is located on chromosome A; gene 2 is located far away but on the same chromosome.
- **Unknown in question:** `situation`
- **Unknown in answer:** `locate`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q73  (tie)

- **Question:** In modern terminology, diversity is understood to be a result of genetic variation. Sources of variation for evolution include all of the following except
- **Correct answer:** mistakes in translation of structural genes.
- **Unknown in question:** `modern, terminology, diversity, understand, include`
- **Unknown in answer:** `structural`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q102  (tie)

- **Question:** Which of the following is NOT a characteristic of asexual reproduction in animals?
- **Correct answer:** The daughter cells fuse to form a zygote.
- **Unknown in answer:** `fuse, zygote`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q117  (abstain)

- **Question:** Which of the following statements is NOT correct about apoptosis?
- **Correct answer:** Apoptosis, a special type of cell division, requires multiple cell signaling.
- **Unknown in question:** `follow, statement, apoptosis`
- **Unknown in answer:** `apoptosis, special, multiple`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q124  (tie)

- **Question:** Females with Turner's syndrome have a high incidence of hemophilia, a recessive, X-linked trait. Based on this information, it can be inferred that females with this condition
- **Correct answer:** lack an X chromosome
- **Unknown in question:** `high, incidence, base, information, infer, condition`
- **Unknown in answer:** `lack`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q132  (wrong)

- **Question:** Regarding meiosis and mitosis, one difference between the two forms of cellular reproduction is that in meiosis
- **Correct answer:** separation of sister chromatids occurs during the second division, whereas in mitosis separation of sister chromatids occurs during the first division
- **Unknown in question:** `regard, difference, cellular`
- **Unknown in answer:** `separation`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q170  (wrong)

- **Question:** Which of the following statements about meiosis is correct?
- **Correct answer:** The number of chromosomes is reduced.
- **Unknown in question:** `follow, statement`
- **Unknown in answer:** `reduce`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q179  (abstain)

- **Question:** Microtubules are protein filaments assembled from tubulin protein subunits. The chemical colchicine prevents the assembly of microtubules by binding to tubulin protein subunits. Which of the following cellular processes would be most impaired by the application of colchicine?
- **Correct answer:** the alignment and separation of chromosomes during mitosis
- **Unknown in question:** `filament, assemble, tubulin, subunit, chemical, follow, cellular, impair, application`
- **Unknown in answer:** `alignment, separation`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q182  (abstain)

- **Question:** Recently, seasonal dead zones in low-oxygen waters have been occurring annually in the Gulf of Mexico near the mouth of the Mississippi River. The dead zones result from the rapid growth of photosynthetic phytoplankton (algal blooms) and their subsequent decay by oxygen-depleting microbes in the water column. Which of the following factors most likely triggers the algal blooms and the associated dead zones?
- **Correct answer:** A summer influx of nutrients derived from chemical fertilizers that are high in nitrogen and phosphorus
- **Unknown in question:** `seasonal, dead, zone, low, oxygen, water, gulf, mexico, mouth, mississippi, river, photosynthetic, phytoplankton, algal, bloom, subsequent, decay, deplete, microbe, column, follow, associated`
- **Unknown in answer:** `summer, influx, nutrient, derive, chemical, fertilizer, high, nitrogen, phosphorus`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_

### Q211  (tie)

- **Question:** All of the following support the endosymbiotic theory that ancestors of mitochondria and chloroplasts were once independent, free-living prokaryotes EXCEPT:
- **Correct answer:** Mitochondria and chloroplasts function independently from the eukaryotic host cell.
- **Unknown in question:** `support, endosymbiotic, theory, ancestor, chloroplast, independent, free, live, except`
- **Unknown in answer:** `chloroplast, host`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q229  (tie)

- **Question:** Which of the following statements best explains why Mendel's principle of segregation was deemed a law?
- **Correct answer:** The patterns of trait inheritance observed in pea plants were repeatedly demonstrated in other eukaryotes.
- **Unknown in question:** `follow, statement, explain, deem`
- **Unknown in answer:** `pattern, inheritance, observe, pea, eukaryote`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q291  (tie)

- **Question:** Compared to DNA viral replication, RNA viral replication is more susceptible to mutation because
- **Correct answer:** RNA replication lacks the proofreading functions characteristic of DNA replication.
- **Unknown in question:** `compare, viral, susceptible`
- **Unknown in answer:** `lack, proofreading`
- **Linking path exists:** yes
- [ ] Teach: _(fact to add)_

### Q305  (wrong)

- **Question:** Which is a true statement concerning genetic variation?
- **Correct answer:** It must be present in a population before natural selection can act upon the population.
- **Unknown in question:** `statement, concern`
- **Unknown in answer:** `present, population, natural, act`
- **Linking path exists:** no
- [ ] Teach: _(fact to add)_
