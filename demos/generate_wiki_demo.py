"""Generate pre-baked Wikipedia demo: Newton's Laws + Solar System.

Facts extracted directly from:
  - https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion
  - https://en.wikipedia.org/wiki/Solar_System

Every source_text in the path records the Wikipedia URL for provenance.
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sara_brain.core.brain import Brain

NEWTON_URL = "https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion"
SOLAR_URL = "https://en.wikipedia.org/wiki/Solar_System"

b = Brain(":memory:")

def teach_batch(facts, source_url):
    count = 0
    for stmt in facts:
        r = b.teach(f"{stmt}")
        if r:
            # Update the path's source_text to include the Wikipedia URL
            b.conn.execute(
                "UPDATE paths SET source_text = ? WHERE id = ?",
                (f"{stmt} [source: {source_url}]", r.path_id)
            )
            count += 1
    b.conn.commit()
    return count

# ── Newton's Laws of Motion ──
# Extracted from the actual Wikipedia article text
newton_facts = [
    # Overview (from opening paragraphs)
    "newton laws of motion is three physical laws",
    "newton laws of motion requires forces",
    "newton laws of motion is basis for newtonian mechanics",
    "isaac newton is author of principia mathematica",
    "principia mathematica is published in 1687",
    "classical mechanics is built on newton laws",

    # Prerequisites (from Prerequisites section)
    "kinematics is mathematical description of motion",
    "velocity is derivative of position",
    "velocity equation is v equals ds over dt",
    "acceleration is derivative of velocity",
    "acceleration equation is a equals dv over dt",
    "acceleration is second derivative of position",
    "position is a vector",
    "velocity is a vector",
    "acceleration is a vector",
    "force is a vector",
    "force is not same as power",
    "force is not same as pressure",
    "mass is not same as weight",

    # First Law (from First law section)
    "first law is principle of inertia",
    "first law requires no net force",
    "inertia is natural behavior to move in straight line",
    "inertia is natural behavior at constant speed",
    "first law is no privileged inertial observer",
    "first law requires no absolute standard of rest",

    # Second Law (from Second law section)
    "momentum is product of mass and velocity",
    "momentum equation is p equals mv",
    "second law is force equals rate of change of momentum",
    "second law equation is F equals dp over dt",
    "force equals mass times acceleration",
    "force equation is F equals ma",
    "second law equation is F equals m times dv over dt",
    "net force is zero means mechanical equilibrium",
    "free body diagram is visual representation of forces",

    # Universal Gravitation (from Uniformly accelerated motion section)
    "gravitational force equation is F equals GMm over r squared",
    "gravitational acceleration equation is g equals GM over r squared",
    "gravitational acceleration is approximately 9.8 m per s squared",
    "free fall is acceleration at constant rate",
    "free fall acceleration is same for all bodies",
    "projectile motion follows parabola shaped trajectory",
    "gravity affects vertical motion not horizontal",

    # Circular Motion (from Uniform circular motion section)
    "centripetal acceleration equation is a equals v squared over r",
    "centripetal force is directed toward center of circle",
    "centripetal force equation is mv squared over r",
    "moon orbit is approximated by uniform circular motion",
    "centripetal force is gravity for orbits",

    # Third Law (from Third law section)
    "third law is action and reaction",
    "third law requires equal magnitude opposite direction",
    "third law requires forces on different bodies",
    "third law is related to conservation of momentum",
    "momentum is conserved",
    "total momentum is p1 plus p2",

    # Harmonic Motion (from Harmonic motion section)
    "simple harmonic motion equation is F equals negative kx",
    "harmonic oscillator frequency is square root of k over m",
    "pendulum is example of harmonic oscillator",
    "pendulum frequency equation is omega equals square root of g over L",
    "resonance is driven harmonic oscillator phenomenon",

    # Work and Energy (from Work and energy section)
    "energy is classified into kinetic and potential",
    "kinetic energy is due to motion",
    "potential energy is due to position",
    "thermal energy is kinetic energy of atoms and molecules",
    "work energy theorem is work equals change in kinetic energy",
    "force equals negative gradient of potential",
    "force equation is F equals negative gradient U",
    "energy is conserved",

    # Rotation (from Rigid-body motion section)
    "moment of inertia is analogue of mass for rotation",
    "angular momentum is analogue of momentum for rotation",
    "torque is analogue of force for rotation",
    "angular momentum equation is L equals r cross p",
    "torque equation is tau equals r cross F",
    "angular momentum is conserved when torque is zero",

    # Gravitation and Kepler (from Multi-body gravitational system section)
    "universal gravitation is force proportional to product of masses",
    "universal gravitation is force inversely proportional to distance squared",
    "orbits are conic sections",
    "orbits includes ellipses",
    "orbits includes parabolas",
    "orbits includes hyperbolas",
    "planets have elliptical orbits",
    "kepler problem is finding orbit shape from inverse square force",
    "three body problem has no exact closed form solution",

    # Lagrangian (from Lagrangian section)
    "lagrangian is difference of kinetic and potential energy",
    "lagrangian equation is L equals T minus V",
    "kinetic energy equation is T equals half mv squared",
    "euler lagrange equation is foundation of lagrangian mechanics",
    "noether theorem relates symmetries and conservation laws",

    # Hamiltonian (from Hamiltonian section)
    "hamiltonian is often equal to total energy",
    "hamiltonian equation is H equals p squared over 2m plus V",
    "hamilton equations gives time derivatives of position and momentum",

    # Electromagnetism (from Electromagnetism section)
    "coulomb law is similar form to gravitational law",
    "coulomb force is proportional to product of charges",
    "coulomb force is inversely proportional to distance squared",
    "lorentz force equation is F equals qE plus qv cross B",

    # Special Relativity (from Special relativity section)
    "special relativity is revision of space and time",
    "relativistic momentum equation is p equals m gamma v",
    "speed of light is maximum speed",
    "newtonian mechanics is approximation for slow speeds",

    # General Relativity (from General relativity section)
    "general relativity is gravity as curvature of spacetime",
    "spacetime tells matter how to move",
    "matter tells spacetime how to curve",
    "einstein field equations requires tensor calculus",

    # Chaos (from Chaos section)
    "newton laws allows chaos",
    "sensitive dependence on initial conditions is chaos",
    "double pendulum is example of chaos",
    "navier stokes equation is newton second law for fluids",

    # History (from History section)
    "aristotle is predecessor of newton",
    "galileo is credited with concept of inertia",
    "descartes introduced concept of inertia",
    "huygens studied collisions between hard spheres",
    "huygens deduced conservation of momentum",
    "euler pioneered study of rigid bodies",
    "euler established basic theory of fluid dynamics",
    "jakob hermann wrote F equals ma in 1716",
]

# ── Solar System ──
# Extracted from the actual Wikipedia article text
solar_facts = [
    # Overview (from opening paragraphs)
    "solar system is gravitationally bound system",
    "solar system contains sun",
    "solar system contains eight planets",
    "solar system is formed 4.6 billion years ago",
    "solar system is formed from molecular cloud collapse",
    "sun is 99.86 percent of solar system mass",
    "sun fuses hydrogen into helium",

    # Planets overview
    "mercury is a terrestrial planet",
    "venus is a terrestrial planet",
    "earth is a terrestrial planet",
    "mars is a terrestrial planet",
    "jupiter is a gas giant",
    "saturn is a gas giant",
    "uranus is an ice giant",
    "neptune is an ice giant",
    "jupiter and saturn is 90 percent of non stellar mass",

    # Inner planets (from Inner planets section)
    "mercury is smallest planet",
    "mercury has widely varying temperature",
    "mercury has no natural satellites",
    "venus has atmosphere of carbon dioxide",
    "venus has surface temperature over 400 celsius",
    "venus has no natural satellites",
    "earth is only place with known life",
    "earth has atmosphere of 78 percent nitrogen",
    "earth has atmosphere of 21 percent oxygen",
    "earth has plate tectonics",
    "earth has magnetic field",
    "moon is earth only natural satellite",
    "moon has diameter one quarter of earth",
    "mars has radius half of earth",
    "mars is red due to iron oxide",
    "mars has atmosphere of carbon dioxide",
    "mars has two moons",
    "phobos is mars inner moon",
    "deimos is mars outer moon",

    # Orbits (from Orbits section)
    "kepler laws describes orbits of objects around sun",
    "orbit is ellipse with sun at one focus",
    "perihelion is closest approach to sun",
    "aphelion is most distant point from sun",
    "planet orbits are nearly circular",
    "angular momentum is measure of orbital and rotational momentum",
    "sun has only 2 percent of angular momentum",
    "jupiter has most of angular momentum",

    # Outer planets (from Outer planets section)
    "jupiter is biggest planet",
    "jupiter is most massive planet",
    "jupiter has great red spot",
    "jupiter has magnetosphere",
    "jupiter has 101 confirmed satellites",
    "jupiter has faint ring",
    "ganymede is largest moon of jupiter",
    "saturn has visible ring system",
    "saturn has rings of ice and rock particles",
    "saturn has hexagon shaped storms",
    "saturn has 285 confirmed satellites",
    "titan is only satellite with substantial atmosphere",
    "uranus has axial tilt greater than 90 degrees",
    "uranus has 29 confirmed satellites",
    "neptune is furthest known planet",
    "neptune has 16 confirmed satellites",
    "triton is neptune largest moon",
    "triton has erupting geysers of nitrogen",

    # Asteroid belt (from Asteroid belt section)
    "asteroid belt is between mars and jupiter orbits",
    "asteroid belt is remnants from solar system formation",
    "ceres is only dwarf planet in asteroid belt",
    "ceres has diameter of 940 km",
    "vesta is second largest in asteroid belt",

    # Formation (from Formation section)
    "solar system formed from gravitational collapse",
    "protoplanetary disc has diameter of roughly 200 AU",
    "planets formed by accretion",
    "frost line is between mars and jupiter",
    "giant planets formed beyond frost line",
    "solar wind created heliosphere",

    # Kuiper belt (from Kuiper belt section)
    "kuiper belt is between 30 and 50 AU from sun",
    "kuiper belt is composed mainly of ice objects",
    "pluto is largest known object in kuiper belt",
    "pluto has 2 to 3 resonance with neptune",
    "pluto has five moons",
    "charon is largest moon of pluto",

    # Comets (from Comets section)
    "comets is composed of volatile ices",
    "comets has highly eccentric orbits",
    "short period comets is from kuiper belt",
    "long period comets is from oort cloud",
    "coma is tail of gas and dust",

    # Oort cloud (from Oort cloud section)
    "oort cloud is theorized spherical shell",
    "oort cloud is source for long period comets",
    "oort cloud is from 2000 AU to 200000 AU",

    # Gravitational concepts (overlap with Newton!)
    "gravity is force binding solar system",
    "gravity requires mass",
    "orbit requires gravity",
    "orbit requires velocity",
    "escape velocity requires gravity",
    "tidal force requires gravity",
    "tidal force requires distance",
    "gravitational collapse requires mass",
    "orbital period requires distance from sun",

    # Sun details (from Sun section)
    "sun is G type main sequence star",
    "sun has mass of 332900 earth masses",
    "sun has nuclear fusion of hydrogen",
    "sun has heliosphere",
    "sun has solar wind",
    "solar wind has speed up to 2880000 km per hour",
    "solar flares disturbs heliosphere",
    "heliospheric current sheet is largest stable structure",

    # Distances (from Distances section)
    "jupiter is 5.2 AU from sun",
    "neptune is 30 AU from sun",
    "astronomical unit is earth sun distance",

    # Discovery (from Discovery section)
    "copernicus developed heliocentric system",
    "kepler produced rudolphine tables",
    "kepler allowed orbits to be elliptical",
    "galileo discovered jupiter has four satellites",
    "huygens discovered titan",
    "huygens discovered shape of saturn rings",
    "newton demonstrated same laws apply on earth and in skies",
    "halley realized comet returns every 75 to 76 years",
    "uranus is recognized as planet by 1783",
    "neptune is identified as planet in 1846",
]

print("=== Teaching Newton's Laws of Motion ===")
print(f"    Source: {NEWTON_URL}")
nc = teach_batch(newton_facts, NEWTON_URL)
print(f"    Taught {nc}/{len(newton_facts)} facts")

print()
print("=== Teaching Solar System ===")
print(f"    Source: {SOLAR_URL}")
sc = teach_batch(solar_facts, SOLAR_URL)
print(f"    Taught {sc}/{len(solar_facts)} facts")

print()
stats = b.stats()
print(f"=== Brain Stats ===")
print(f"    Neurons:  {stats['neurons']}")
print(f"    Segments: {stats['segments']}")
print(f"    Paths:    {stats['paths']}")

# Show cross-domain connections
print()
print("=== Cross-Domain: trace gravity ===")
for t in b.trace("gravity"):
    labels = " -> ".join(n.label for n in t.neurons)
    print(f"    {labels}")

print()
print("=== Cross-Domain: recognize gravity, mass, orbit ===")
results = b.recognize("gravity, mass, orbit")
for r in results[:10]:
    print(f"    {r.neuron.label} ({r.confidence} converging paths)")

print()
print("=== Equation lookup: why force ===")
for t in b.why("force"):
    labels = " -> ".join(n.label for n in t.neurons)
    src = t.source_text or ""
    print(f"    {labels}")
    if src:
        print(f"      source: {src}")

# Export brain state
neurons = b.neuron_repo.list_all()
segments = b.segment_repo.list_all()
all_paths = b.path_repo.list_all()

data = {
    "version": 1,
    "source_urls": [NEWTON_URL, SOLAR_URL],
    "attribution": "Facts extracted from Wikipedia articles. Content licensed under CC BY-SA 4.0.",
    "neurons": [
        {"id": n.id, "label": n.label, "neuron_type": n.neuron_type.value,
         "created_at": n.created_at}
        for n in neurons
    ],
    "segments": [
        {"id": s.id, "source_id": s.source_id, "target_id": s.target_id,
         "relation": s.relation, "strength": s.strength,
         "traversals": s.traversals, "created_at": s.created_at,
         "last_used": s.last_used}
        for s in segments
    ],
    "paths": [
        {"id": p.id, "origin_id": p.origin_id, "terminus_id": p.terminus_id,
         "source_text": p.source_text, "created_at": p.created_at}
        for p in all_paths
    ],
    "path_steps": [],
}

for p in all_paths:
    steps = b.path_repo.get_steps(p.id)
    for step in steps:
        data["path_steps"].append({
            "id": step.id,
            "path_id": step.path_id,
            "step_order": step.step_order,
            "segment_id": step.segment_id,
        })

with open("demos/wiki_demo_brain.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print()
print(f"=== Exported to demos/wiki_demo_brain.json ===")
print(f"    {len(data['neurons'])} neurons, {len(data['segments'])} segments, "
      f"{len(data['paths'])} paths, {len(data['path_steps'])} steps")

b.close()
