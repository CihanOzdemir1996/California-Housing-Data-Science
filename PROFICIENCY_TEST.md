# 🎯 California Housing Data Science - Proficiency Test

## Test Overview
This proficiency test evaluates your understanding of the California Housing Data Science project, covering data science concepts, Python programming, machine learning, and code quality.

**Time Limit:** 90 minutes  
**Total Points:** 100 points

---

## Part 1: Code Understanding & Analysis (25 points)

### Question 1.1 (5 points)
In `california_housing_prediction.py`, the code uses `StandardScaler` but doesn't actually use the scaled data for training. Why is this acceptable for Random Forest, and when would you need to use scaling?

**Answer Space:**
```
[Your answer here - explain the concept]
```

---

### Question 1.2 (5 points)
The Streamlit app (`app.py`) has hardcoded values for `AveBedrms` and `Population` in the prediction input. What are the potential issues with this approach, and how would you fix it?

**Answer Space:**
```
[Your answer here]
```

---

### Question 1.3 (5 points)
Explain the difference between the metrics used in the evaluation section (MSE, RMSE, MAE, R²). When would you prefer one over the others?

**Answer Space:**
```
[Your answer here]
```

---

### Question 1.4 (5 points)
The code uses `@st.cache_resource` decorator in the Streamlit app. What does this do, and why is it important for this application?

**Answer Space:**
```
[Your answer here]
```

---

### Question 1.5 (5 points)
In the EDA section, the code detects outliers using the IQR method but doesn't remove them. Should outliers be removed for this dataset? Justify your answer.

**Answer Space:**
```
[Your answer here]
```

---

## Part 2: Code Review & Improvements (25 points)

### Question 2.1 (10 points)
**Code Review Task:** Review the `app.py` file and identify at least 5 issues or improvements. Provide specific line numbers and explanations.

**Issues Found:**
1. [Line X]: [Issue description]
2. [Line Y]: [Issue description]
3. [Line Z]: [Issue description]
4. [Line W]: [Issue description]
5. [Line V]: [Issue description]

---

### Question 2.2 (10 points)
**Improvement Task:** The current code doesn't persist the trained model. Write a solution to:
- Save the trained model after training
- Load the saved model in the Streamlit app instead of retraining

**Your Solution:**
```python
# Add your code here
```

---

### Question 2.3 (5 points)
The matplotlib style `'seaborn-v0_8-darkgrid'` is used but may be deprecated. What would be a better approach for styling plots?

**Answer Space:**
```
[Your answer here]
```

---

## Part 3: Machine Learning Concepts (20 points)

### Question 3.1 (5 points)
Why is Random Forest a good choice for this regression problem? What are its advantages and potential disadvantages?

**Answer Space:**
```
[Your answer here]
```

---

### Question 3.2 (5 points)
The model uses cross-validation. Explain what 5-fold cross-validation does and why it's important for model evaluation.

**Answer Space:**
```
[Your answer here]
```

---

### Question 3.3 (5 points)
Feature importance is calculated using `rf_model.feature_importances_`. How does Random Forest calculate feature importance? Is this always reliable?

**Answer Space:**
```
[Your answer here]
```

---

### Question 3.4 (5 points)
The model has these hyperparameters: `n_estimators=100`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=2`. Explain what each does and suggest how you would optimize them.

**Answer Space:**
```
[Your answer here]
```

---

## Part 4: Data Science Best Practices (15 points)

### Question 4.1 (5 points)
The code doesn't have any error handling. What errors could occur during runtime, and how would you handle them?

**Answer Space:**
```
[Your answer here]
```

---

### Question 4.2 (5 points)
The dataset is split 80/20 for train/test. Is this appropriate? What factors should be considered when choosing the split ratio?

**Answer Space:**
```
[Your answer here]
```

---

### Question 4.3 (5 points)
The code generates multiple visualizations but doesn't validate if the data meets assumptions (e.g., normality, linearity). What assumptions should be checked for a regression problem, and how?

**Answer Space:**
```
[Your answer here]
```

---

## Part 5: Practical Coding Challenge (15 points)

### Question 5.1 (15 points)
**Task:** Write a function that:
1. Takes a trained Random Forest model and test data as input
2. Calculates prediction intervals (not just point predictions)
3. Returns predictions with confidence intervals (e.g., 95% confidence interval)

**Your Solution:**
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict_with_intervals(model, X_test, confidence=0.95):
    """
    Predict with confidence intervals using Random Forest.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained Random Forest model
    X_test : array-like
        Test features
    confidence : float
        Confidence level (default 0.95)
    
    Returns:
    --------
    predictions : array
        Point predictions
    lower_bound : array
        Lower bound of confidence interval
    upper_bound : array
        Upper bound of confidence interval
    """
    # Your implementation here
    pass
```

---

## Part 6: Advanced Concepts (Optional - Bonus 10 points)

### Question 6.1 (5 points - Bonus)
How would you implement feature engineering to potentially improve model performance? Give 3 specific examples for this dataset.

**Answer Space:**
```
[Your answer here]
```

---

### Question 6.2 (5 points - Bonus)
The current model doesn't handle missing values (though the dataset has none). How would you modify the code to handle missing values if they existed?

**Answer Space:**
```
[Your answer here]
```

---

## Scoring Rubric

| Section | Points | Criteria |
|---------|--------|----------|
| Part 1: Code Understanding | 25 | Accuracy of explanations, depth of understanding |
| Part 2: Code Review | 25 | Quality of identified issues, practicality of solutions |
| Part 3: ML Concepts | 20 | Correctness of ML theory, understanding of algorithms |
| Part 4: Best Practices | 15 | Awareness of best practices, practical knowledge |
| Part 5: Coding Challenge | 15 | Code correctness, efficiency, completeness |
| Part 6: Advanced (Bonus) | 10 | Advanced problem-solving, creativity |

**Total: 100 points (+ 10 bonus)**

---

## Evaluation Criteria

- **90-100 points:** Excellent - Strong understanding, production-ready code
- **75-89 points:** Good - Solid understanding, minor improvements needed
- **60-74 points:** Satisfactory - Basic understanding, needs more practice
- **Below 60 points:** Needs Improvement - Fundamental concepts need review

---

## Notes for Interviewer

This test evaluates:
1. **Technical Knowledge:** Understanding of ML concepts, Python, and data science
2. **Code Quality Awareness:** Ability to identify issues and suggest improvements
3. **Problem-Solving:** Practical coding skills and solution design
4. **Best Practices:** Knowledge of industry standards and best practices

The candidate should demonstrate:
- Clear understanding of the codebase
- Ability to identify and fix issues
- Knowledge of ML theory and practice
- Writing clean, maintainable code
- Awareness of production considerations
