# AG News Classification - Multinomial Naive Bayes Evaluation Report

## 1. Dataset & Model Summary
- **Training Dataset Size**: 120,000 articles
- **Test Dataset Size**: 7,600 articles
- **Vocabulary Size**: 30,000 tokens
- **Training Time**: 0.044 seconds
- **Prediction Time (Test Split)**: 0.002 seconds
- **Throughput**: 3795750.2 articles/second

## 2. Overall Performance Metrics
- **Test Accuracy**: 90.18%
- **Macro-Averaged F1**: 90.14%
- **Weighted-Averaged F1**: 90.14%

## 3. Per-Class Performance Breakdown
| Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
| :--- | :---: | :---: | :---: | :---: |
| World | 90.89 | 89.84 | 90.37 | 1,900 |
| Sports | 94.91 | 98.11 | 96.48 | 1,900 |
| Business | 87.57 | 84.53 | 86.02 | 1,900 |
| Sci/Tech | 87.16 | 88.26 | 87.71 | 1,900 |

## 4. Confusion Matrix
| True \ Pred | World | Sports | Business | Sci/Tech |
| :--- | :---: | :---: | :---: | :---: |
| **World** | 1,707 | 60 | 91 | 42 |
| **Sports** | 20 | 1,864 | 8 | 8 |
| **Business** | 78 | 19 | 1,606 | 197 |
| **Sci/Tech** | 73 | 21 | 129 | 1,677 |

## 5. Classification Report (Scikit-Learn Standard Format)
```text
precision    recall  f1-score   support

       World     0.9089    0.8984    0.9037      1900
      Sports     0.9491    0.9811    0.9648      1900
    Business     0.8757    0.8453    0.8602      1900
    Sci/Tech     0.8716    0.8826    0.8771      1900

    accuracy                         0.9018      7600
   macro avg     0.9013    0.9018    0.9014      7600
weighted avg     0.9013    0.9018    0.9014      7600
```

## 6. Qualitative Analysis: Correctly Classified Articles
### Example 1 (Index: 0)
- **Article Text**: "Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
- **Ground Truth**: `Business`
- **Prediction**: `Business` (Confidence: 100.00%)
- **Class Probabilities**: {'World': 3.5788358462253287e-06, 'Sports': 6.990648856948272e-15, 'Business': 0.9999964201283587, 'Sci/Tech': 1.035788135966166e-09}

### Example 2 (Index: 1)
- **Article Text**: "The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com) SPACE.com - TORONTO, Canada -- A second\team of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately funded suborbital space flight, has officially announced the first\launch date for its manned rocket."
- **Ground Truth**: `Sci/Tech`
- **Prediction**: `Sci/Tech` (Confidence: 100.00%)
- **Class Probabilities**: {'World': 1.2234480486078345e-22, 'Sports': 1.65300115668348e-25, 'Business': 4.2720629833156875e-23, 'Sci/Tech': 1.0}

### Example 3 (Index: 2)
- **Article Text**: "Ky. Company Wins Grant to Study Peptides (AP) AP - A company founded by a chemistry researcher at the University of Louisville won a grant to develop a method of producing better peptides, which are short chains of amino acids, the building blocks of proteins."
- **Ground Truth**: `Sci/Tech`
- **Prediction**: `Sci/Tech` (Confidence: 100.00%)
- **Class Probabilities**: {'World': 4.572995401043365e-12, 'Sports': 5.762251752528719e-11, 'Business': 2.0434269250217686e-09, 'Sci/Tech': 0.9999999978943777}

### Example 4 (Index: 3)
- **Article Text**: "Prediction Unit Helps Forecast Wildfires (AP) AP - It's barely dawn when Mike Fitzpatrick starts his shift with a blur of colorful maps, figures and endless charts, but already he knows what the day will bring. Lightning will strike in places he expects. Winds will pick up, moist places will dry and flames will roar."
- **Ground Truth**: `Sci/Tech`
- **Prediction**: `Sci/Tech` (Confidence: 95.07%)
- **Class Probabilities**: {'World': 5.507490872723601e-07, 'Sports': 0.04928394946748686, 'Business': 2.0726187975910132e-08, 'Sci/Tech': 0.950715479057238}

### Example 5 (Index: 4)
- **Article Text**: "Calif. Aims to Limit Farm-Related Smog (AP) AP - Southern California's smog-fighting agency went after emissions of the bovine variety Friday, adopting the nation's first rules to reduce air pollution from dairy cow manure."
- **Ground Truth**: `Sci/Tech`
- **Prediction**: `Sci/Tech` (Confidence: 99.99%)
- **Class Probabilities**: {'World': 5.2396662836340934e-05, 'Sports': 9.590035513142135e-14, 'Business': 1.0968816165910721e-05, 'Sci/Tech': 0.999936634520902}

## 7. Qualitative Analysis: Misclassified Articles
### Error 1 (Index: 19)
- **Article Text**: "Storage, servers bruise HP earnings update Earnings per share rise compared with a year ago, but company misses analysts' expectations by a long shot."
- **Ground Truth**: `Sci/Tech` (Confidence: 2.03%)
- **Predicted Class**: `Business` (Confidence: 97.97%)
- **Class Probabilities**: {'World': 8.946704952798974e-15, 'Sports': 8.131613180222353e-16, 'Business': 0.9797286570570083, 'Sci/Tech': 0.020271342942981933}

### Error 2 (Index: 23)
- **Article Text**: "Some People Not Eligible to Get in on Google IPO Google has billed its IPO as a way for everyday people to get in on the process, denying Wall Street the usual stranglehold it's had on IPOs. Public bidding, a minimum of just five shares, an open process with 28 underwriters - all this pointed to a new level of public participation. But this isn't the case."
- **Ground Truth**: `Sci/Tech` (Confidence: 32.26%)
- **Predicted Class**: `Business` (Confidence: 67.74%)
- **Class Probabilities**: {'World': 4.705862965627123e-11, 'Sports': 1.7079232748020723e-17, 'Business': 0.6773535475100046, 'Sci/Tech': 0.32264645244293666}

### Error 3 (Index: 24)
- **Article Text**: "Rivals Try to Turn Tables on Charles Schwab By MICHAEL LIEDTKE     SAN FRANCISCO (AP) -- With its low prices and iconoclastic attitude, discount stock broker Charles Schwab Corp. (SCH) represented an annoying stone in Wall Street's wing-tipped shoes for decades..."
- **Ground Truth**: `Sci/Tech` (Confidence: 0.00%)
- **Predicted Class**: `Business` (Confidence: 100.00%)
- **Class Probabilities**: {'World': 2.139916898891874e-13, 'Sports': 4.744701649337326e-16, 'Business': 0.999999403542605, 'Sci/Tech': 5.964571804294064e-07}

### Error 4 (Index: 41)
- **Article Text**: "Retailers Vie for Back-To-School Buyers (Reuters) Reuters - Apparel retailers are hoping their\back-to-school fashions will make the grade among\style-conscious teens and young adults this fall, but it could\be a tough sell, with students and parents keeping a tighter\hold on their wallets."
- **Ground Truth**: `Business` (Confidence: 20.59%)
- **Predicted Class**: `Sci/Tech` (Confidence: 79.41%)
- **Class Probabilities**: {'World': 3.065557769739096e-06, 'Sports': 1.1133791319425753e-11, 'Business': 0.20588582373012748, 'Sci/Tech': 0.7941111107009691}

### Error 5 (Index: 47)
- **Article Text**: "Dell Exits Low-End China Consumer PC Market  HONG KONG (Reuters) - Dell Inc. &lt;DELL.O&gt;, the world's  largest PC maker, said on Monday it has left the low-end  consumer PC market in China and cut its overall growth target  for the country this year due to stiff competition in the  segment."
- **Ground Truth**: `Sci/Tech` (Confidence: 0.00%)
- **Predicted Class**: `Business` (Confidence: 100.00%)
- **Class Probabilities**: {'World': 2.95683800580027e-27, 'Sports': 2.2733442088561647e-38, 'Business': 0.9999589440753142, 'Sci/Tech': 4.1055924685943794e-05}
