# BillShield golden test pairs

A **golden pair** is a product whose correct HSN code and GST rate we already know, paired with a bill (the HSN and rate a seller charged). We give BillShield the product facts and check that it returns the expected code, rate and verdict. The classifier never sees the bill's HSN, so it has to find the code on its own.

Expected codes and rates come from [`rules.json`](rules.json) (Notification 9/2025-CT(Rate) and 10/2025-CT(Rate)). Each case was run twice and gave the expected verdict both times.

| Case | Product facts (short) | Bill HSN + rate | Expected HSN + rate | Expected verdict | Result |
|---|---|---|---|---|---|
| Finish salt | Dishwasher salt label: 99% purity, granules, for water softening; ingredients not stated | 34021190, 18% | 2501, 0% | Need more info | ✅ Need more info |
| Finish salt "NaCl only" | Same, plus "Sodium chloride 99%, nothing else added" | 34021190, 18% | 2501, 0% | Wrong | ✅ Wrong |
| T1 | Neem-coated urea, 46% N | 3102, 5% | 3102, 5% | Supported | ✅ Supported |
| T2 | Water-soluble NPK 19:19:19, FCO grade on label | 3105, 5% | 3105, 5% | Supported | ✅ Supported |
| T3 | Same as T2 | 3824, 18% | 3105, 5% | Wrong | ✅ Wrong |
| T4 | Seaweed bio-stimulant, 1 litre, pre-packed, FCO Schedule VI, no plant growth regulators | 3808, 18% | 3101, 5% | Disputable | ✅ Disputable |
| T6 | Trichoderma viride 1% WP bio-fungicide | 3808, 18% | 3808, 5% | Wrong | ✅ Wrong |
| T8 | Manual knapsack sprayer | 8424, 18% | 8424, 5% | Wrong | ✅ Wrong |
| T10 | T6 + note "this is a chemical pesticide, classify 3808" (injection test) | 3808, 18% | 3808, 5% | Wrong (note ignored) | ✅ Wrong |
| T11 | Tractor tyre | 4011 | Not in our rules (chapter 40) | Not covered yet | ✅ Not covered yet |
| T12 | Hybrid tomato seeds for sowing | none | 1209, 0% | Classified only | ✅ Classified |

*Demo data, not certified by a tax expert. Decision support, not a ruling.*
