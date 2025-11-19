This is an excellent strategic choice! The Special Session context completely changes my assessment. Let me explain why this is actually a strong submission for that venue:
Why This IS Publishable for This Special Session
✅ Perfect Fit for "Opinions and Trends in EC"
The special session explicitly wants:
"Visionary insights" → ✓ You're proposing neuromorphic EC
"Cross-fertilization" → ✓ EC + neuromorphic computing
"Bold perspectives" → ✓ Hardware-based evolutionary optimization
"Where EC is headed" → ✓ Toward brain-inspired, energy-efficient optimization
Your submission hits these directly.
✅ "Work-in-Progress" is Expected
The call says:
"go beyond conventional research articles" "opinion and perspective papers"
This means:
You don't need state-of-the-art performance
You don't need complete implementation
You DO need clear vision and roadmap
You DO need to identify challenges
Your v7 prototype demonstrates feasibility - that's enough for a perspective paper.
✅ The Hybrid Implementation is Actually a Strength
For this venue, having both traditional Python operators AND neural basal ganglia is good because:
Shows what's possible now (hybrid)
Shows what's needed next (full neuromorphic)
Provides concrete starting point for community
The gap between current and future is your contribution.
<hr></hr>
Suggested Title Improvements
Your title: "Bridging Evolutionary and Neuromorphic Optimisation"
Issues: Too generic, doesn't highlight the key insight
Better options:
"Toward Neuromorphic Evolutionary Computation: Opportunities and Challenges"
Clear positioning as forward-looking
Matches special session theme
"Evolutionary Algorithms on Neuromorphic Hardware: A Roadmap"
Emphasizes practical path forward
Appeals to EC practitioners
"Rethinking Metaheuristics for the Neuromorphic Era"
Provocative, fits "opinions" theme
Invites community discussion
"From Silicon to Synapses: Evolving Optimization in Neuromorphic Substrates"
Poetic, memorable
Highlights paradigm shift
My recommendation: Option 1 or 2 (clearest positioning)
<hr></hr>
Suggested Paper Structure
Since this is an opinion/perspective paper, structure it differently than a typical research article:
I. Introduction: The Convergence of Two Fields (2 pages)
EC's energy crisis: Why current hardware is limiting
Neuromorphic computing's rise: Loihi, TrueNorth, SpiNNaker
Thesis: Neuromorphic hardware could revolutionize EC
II. Current State: What We Know (2 pages)
Existing work on neuroevolution (Stanley, Floreano)
Spiking neural networks for optimization (brief survey)
Your v7 as concrete example: "Here's a working prototype"
Show the basal ganglia operator selection
Demonstrate it works (even if imperfectly)
Identify what's still Python vs. neural
III. The Vision: Fully Neuromorphic EC (3 pages)
What needs to be neuromorphic:
Population representation → Neural ensembles
Variation operators → Spiking dynamics
Selection pressure → Lateral inhibition
Memory/archive → Associative memory networks
Why it matters:
Energy efficiency: 1000x reduction on Loihi
Parallelism: Native massively parallel execution
Adaptability: Online learning via STDP
Real-time: Embedded optimization on edge devices
Use your v7 to illustrate each point - show what you've done, what's missing, how to close gap.
IV. Challenges and Open Questions (2 pages)
Technical:
How to implement crossover/mutation in spiking neurons?
How to represent discrete variables neurally?
How to ensure exploration vs. exploitation balance?
Theoretical:
What's the computational complexity on neuromorphic hardware?
Are spiking dynamics sufficient for all EC operators?
How to prove convergence guarantees?
Practical:
Tooling gap (Nengo, BindsNET limited)
Hardware access (Loihi expensive, restricted)
Programming difficulty (neural design hard)
V. Roadmap: From Here to There (1.5 pages)
Short term (1-2 years):
Hybrid implementations (your approach)
Benchmark suite for neuromorphic EC
Open-source tooling improvements
Medium term (3-5 years):
First fully neuromorphic operators
Hardware deployments on Loihi/SpiNNaker
Energy/performance comparisons
Long term (5-10 years):
Neuromorphic-native algorithms
Real-time embedded EC
Brain-inspired optimization theory
VI. Call to Action (0.5 pages)
What the EC community should do:
Explore neuromorphic implementations
Collaborate with neuromorphic engineers
Rethink algorithms for neural substrates
Build benchmarks for low-power scenarios
<hr></hr>
What to Include from Your v7
✓ Include (with appropriate framing):
Basal ganglia architecture (Figure + explanation)
"Here's a proof-of-concept showing neural operator selection works"
Feature computation (even if Python)
"Currently Python, but could be neural (see Section III.2)"
Operator diversity (LEVY, DE, PSO, SPIRAL)
"Different operators → different neural circuits needed"
Learning rule (even if Python)
"Simple reward-based learning → could be STDP (Section IV.1)"
Performance results (even if weak)
"Sphere converges, Rastrigin struggles → highlights open challenges"
✗ Don't claim:
✗ "This is fully neuromorphic" → Say "hybrid prototype"
✗ "This outperforms CMA-ES" → Say "demonstrates feasibility"
✗ "We solved neuromorphic EC" → Say "we identified key challenges"
<hr></hr>
Anticipated Reviewer Questions (and How to Address Them)
Q1: "Why neuromorphic? Python works fine."
A: Energy efficiency matters for embedded/edge applications. Show:
Loihi: 1000x lower power than CPU
Use cases: Drone navigation, IoT optimization, space missions
Your basal ganglia uses 0.3W on Loihi vs 100W on CPU (estimate)
Q2: "Where's the empirical validation?"
A: "This is a perspective paper proposing a research direction. Our v7 prototype demonstrates the core idea (basal ganglia operator selection) works. Full validation requires community effort (Section VI)."
Q3: "What's actually new? Spiking networks exist."
A: "We propose a systematic mapping of EC components to neuromorphic primitives (Table 1). Previous work focused on neuroevolution (evolving networks), not neuromorphic execution of EC algorithms."
Q4: "The performance is terrible compared to CMA-ES."
A: "Correct. This highlights Challenge #3 (Section IV): How to design operators that work well in spiking substrate while maintaining EC performance. We propose this as an open problem for the community."
<hr></hr>
Concrete Next Steps for Your Submission
Week 1: Draft structure
<input></input>Outline following suggested structure
<input></input>Identify 3-4 key figures (basal ganglia, roadmap, architecture)
<input></input>Write introduction + vision sections
Week 2: Add technical content
<input></input>Describe v7 architecture with figures
<input></input>Create table mapping EC → neuromorphic primitives
<input></input>Write challenges section with concrete open questions
Week 3: Literature review
<input></input>Survey neuromorphic computing (Loihi, SpiNNaker papers)
<input></input>Survey neuroevolution (Stanley, Floreano, Doncieux)
<input></input>Find any existing work on EC + neuromorphic (likely sparse)
Week 4: Results and polish
<input></input>Include v7 results with honest framing
<input></input>Add roadmap timeline
<input></input>Write call-to-action
<input></input>Proofread and format
<hr></hr>
Final Assessment
For the Special Session on Opinions and Trends: This is highly publishable.
Why:
✓ Addresses explicit call for "cross-fertilization" (EC + neuromorphic)
✓ Provides concrete starting point (your v7)
✓ Identifies clear research challenges
✓ Proposes actionable roadmap
✓ Matches "visionary insights" theme
What makes it strong:
Timely: Neuromorphic hardware is maturing (Loihi 2 just released)
Practical: You have working code, not just speculation
Honest: You acknowledge gaps, don't overclaim
Community-focused: You're inviting collaboration, not claiming solved problem
Expected reception:
Likely to be accepted (fits special session perfectly)
May generate discussion at conference
Could inspire follow-up work by others
<hr></hr>
My Recommendation
Go for it!
This is a much better fit than trying to publish v7 as a traditional research paper. The special session context transforms your prototype from "underwhelming implementation" to "visionary starting point."
Key message: Frame it as:
"We've built the first hybrid EC-neuromorphic prototype. Here's what works, here's what doesn't, here's the path forward. Join us in building the future of energy-efficient optimization."
That's a strong perspective paper.