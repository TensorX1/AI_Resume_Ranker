# Imports
import json
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# AdvancedAnalytics Class
class AdvancedAnalytics:
    """
    A class to perform advanced analytics on hiring data.

    This class provides a suite of tools to analyze hiring patterns, build
    predictive models for candidate success, and generate actionable insights
    to improve the recruitment process.
    """

    def __init__(self):
        """Initializes the AdvancedAnalytics class with empty data structures."""
        self.hiring_data = []
        self.skill_weights = {}
        self.performance_metrics = {}
        self.pattern_analysis = {}
        self.best_model_details = {}

    # 1. Core Workflow Methods

    def load_hiring_data(self, hiring_data_file='hiring_data.json'):
        """
        Loads hiring decision data from a JSON file.

        Args:
            hiring_data_file (str): The path to the JSON file containing hiring records.

        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        try:
            with open(hiring_data_file, 'r', encoding='utf-8') as f:
                self.hiring_data = json.load(f)
            print(f"✅ Loaded {len(self.hiring_data)} hiring records successfully.")
            return True
        except FileNotFoundError:
            print(f"❌ Error: Hiring data file not found at '{hiring_data_file}'.")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred while loading data: {e}")
            return False

    def analyze_hiring_patterns(self):
        """
        Analyzes patterns in hiring data to identify key success factors.

        This method calculates overall hiring rates, analyzes the success rates of
        different skills, and compares the average experience and similarity scores
        between hired and not-hired candidates.

        Returns:
            dict: A dictionary containing the detailed analysis results.
        """
        if not self.hiring_data:
            return {"error": "No hiring data available to analyze."}

        df = pd.DataFrame(self.hiring_data)
        hired_df = df[df['hired'] == True]
        not_hired_df = df[df['hired'] == False]

        # --- Overall hiring statistics ---
        analysis = {
            'total_candidates': len(df),
            'hired_count': len(hired_df),
            'not_hired_count': len(not_hired_df),
            'hiring_rate': len(hired_df) / len(df) if len(df) > 0 else 0
        }

        #   Skill success rate analysis
        all_skills = [skill for sublist in df['skills'] for skill in sublist]
        hired_skills = [skill for sublist in hired_df['skills'] for skill in sublist]

        skill_total_counts = Counter(all_skills)
        skill_hired_counts = Counter(hired_skills)

        skill_success_rates = {
            skill: {
                'success_rate': skill_hired_counts.get(skill, 0) / total,
                'total_appearances': total,
                'hired_appearances': skill_hired_counts.get(skill, 0)
            } for skill, total in skill_total_counts.items()
        }

        sorted_skills = sorted(
            skill_success_rates.items(),
            key=lambda item: item[1]['success_rate'],
            reverse=True
        )
        analysis['skill_analysis'] = {
            'top_successful_skills': sorted_skills[:10],
            'skill_success_rates': skill_success_rates
        }

        #  Experience and Similarity analysis
        if not hired_df.empty and not not_hired_df.empty:
            analysis['experience_analysis'] = {
                'avg_experience_hired': hired_df['experience_years'].mean(),
                'avg_experience_not_hired': not_hired_df['experience_years'].mean()
            }
            analysis['similarity_analysis'] = {
                'avg_similarity_hired': hired_df['similarity_score'].mean(),
                'avg_similarity_not_hired': not_hired_df['similarity_score'].mean()
            }

        self.pattern_analysis = analysis
        return analysis

    def calculate_skill_weights(self):
        """
        Calculates weights for each skill based on hiring success rate and frequency.

        The weight for each skill is determined by its success rate, amplified by
        the logarithmic frequency of its appearance. This balances the importance
        of skills that lead to hires with how common those skills are. The final
        weights are normalized to a 0-1 scale.

        Returns:
            dict: A dictionary mapping each skill to its calculated normalized weight.
        """
        if not self.pattern_analysis.get('skill_analysis'):
            return {}

        skill_rates = self.pattern_analysis['skill_analysis']['skill_success_rates']
        skill_weights = {}

        for skill, data in skill_rates.items():
            success_rate = data['success_rate']
            frequency = data['total_appearances']

            # The log term dampens the effect of very frequent skills,
            # to prevent them from dominating the weight.
            weight = success_rate * (1 + np.log(frequency + 1))
            skill_weights[skill] = weight

        # Normalize weights to a 0-1 scale for consistency
        if skill_weights:
            max_weight = max(skill_weights.values())
            if max_weight > 0:
                self.skill_weights = {k: v / max_weight for k, v in skill_weights.items()}

        return self.skill_weights

    def build_hiring_prediction_model(self):
        """
        Builds and evaluates multiple machine learning models to predict hiring success.

        It trains Logistic Regression, Random Forest, and Gradient Boosting models,
        evaluates them on a test set, and selects the best one based on the F1-score.

        Returns:
            dict: A dictionary containing performance metrics for all models
                  and details of the best-performing model.
        """
        if len(self.hiring_data) < 10:
            return {"error": "Insufficient data to build a reliable model."}

        df = pd.DataFrame(self.hiring_data)
        if len(df['hired'].unique()) < 2:
            return {"error": "Training data must contain both hired and not hired examples."}

        #  Feature Engineering
        skill_feature_list = self._get_skill_feature_list()
        features = [self._vectorize_candidate(row, skill_feature_list) for _, row in df.iterrows()]
        labels = df['hired'].astype(int).values

        X = np.array(features)
        y = np.array(labels)

        #  Data Splitting and Scaling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #  Model Training and Evaluation
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        }

        model_results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            model_results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
            }

        #  Select Best Model
        best_model_name = max(model_results, key=lambda name: model_results[name]['f1_score'])

        self.best_model_details = {
            'model': models[best_model_name],
            'scaler': scaler,
            'name': best_model_name
        }

        self.performance_metrics = {
            'best_model': best_model_name,
            'performance': model_results
        }
        return self.performance_metrics

    def predict_hiring_success(self, candidate_data):
        """
        Predicts the hiring likelihood for a new candidate using the best-trained model.

        Args:
            candidate_data (dict): A dictionary with the new candidate's details.

        Returns:
            dict: A dictionary containing the prediction and the probability of being hired.
        """
        if not self.best_model_details:
            return {"error": "No trained model available. Please run `build_hiring_prediction_model` first."}

        #  Vectorize candidate data using the same feature set as training
        skill_feature_list = self._get_skill_feature_list()
        feature_vector = self._vectorize_candidate(candidate_data, skill_feature_list)

        #  Scale and Predict
        scaler = self.best_model_details['scaler']
        model = self.best_model_details['model']

        X_scaled = scaler.transform(np.array([feature_vector]))
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        return {
            'predicted_hired': bool(prediction),
            'hiring_probability': float(probability[1]),
            'model_used': self.best_model_details['name']
        }

    # 2. Reporting and Visualization Methods

    def generate_performance_report(self):
        """
        Generates a comprehensive report of the entire analysis.

        The report includes model performance metrics, pattern analysis insights,
        calculated skill weights, and actionable recommendations.

        Returns:
            dict: A dictionary containing the full analysis report.
        """
        if not self.performance_metrics:
            return {"error": "No performance metrics available. Run model building first."}

        return {
            'model_performance': self.performance_metrics,
            'pattern_analysis': self.pattern_analysis,
            'skill_weights': self.skill_weights,
            'recommendations': self._generate_recommendations()
        }

    def create_visualizations(self, save_path='analytics_visualizations.png'):
        """
        Creates and saves a dashboard of visualizations from the analysis results.

        Args:
            save_path (str): The file path to save the generated image.

        Returns:
            dict: A success message with the save path or an error message.
        """
        if not self.pattern_analysis:
            return {"error": "No analysis data available to create visualizations."}

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        sns.set_style("whitegrid")
        fig.suptitle('Hiring Analytics Dashboard', fontsize=20)

        # Plot 1: Top 10 Skills by Success Rate
        if 'skill_analysis' in self.pattern_analysis:
            top_skills_data = self.pattern_analysis['skill_analysis']['top_successful_skills']
            skills = [s[0] for s in top_skills_data]
            rates = [s[1]['success_rate'] for s in top_skills_data]
            sns.barplot(x=rates, y=skills, ax=axes[0, 0], palette='viridis')
            axes[0, 0].set_title('Top 10 Skills by Hiring Success Rate')
            axes[0, 0].set_xlabel('Success Rate')

        # Plot 2: Average Experience by Hiring Status
        if 'experience_analysis' in self.pattern_analysis:
            exp_data = self.pattern_analysis['experience_analysis']
            sns.barplot(x=['Hired', 'Not Hired'], y=exp_data.values(), ax=axes[0, 1], palette='coolwarm')
            axes[0, 1].set_title('Average Experience (Years)')
            axes[0, 1].set_ylabel('Years')

        # Plot 3: Average Similarity Score by Hiring Status
        if 'similarity_analysis' in self.pattern_analysis:
            sim_data = self.pattern_analysis['similarity_analysis']
            sns.barplot(x=['Hired', 'Not Hired'], y=sim_data.values(), ax=axes[1, 0], palette='coolwarm')
            axes[1, 0].set_title('Average CV Similarity Score')
            axes[1, 0].set_ylabel('Similarity Score')

        # Plot 4: Model Performance Comparison
        if 'performance' in self.performance_metrics:
            perf_data = self.performance_metrics['performance']
            models = list(perf_data.keys())
            f1_scores = [perf_data[m]['f1_score'] for m in models]
            sns.barplot(x=models, y=f1_scores, ax=axes[1, 1], palette='plasma')
            axes[1, 1].set_title('Model Performance (F1-Score)')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].tick_params(axis='x', rotation=15)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300)
        plt.close()

        return {"message": f"Visualizations saved to {save_path}"}

    # 3. Clustering Methods

    def get_optimal_clustering(self, data):
        """
        Automatically selects and applies the best clustering algorithm to the data.

        It chooses between K-Means and DBSCAN based on data size and performance,
        then returns the clustering results.

        Args:
            data (list or pd.DataFrame): The dataset to cluster.

        Returns:
            dict: The clustering results including the algorithm used, number of
                  clusters, silhouette score, and cluster labels.
        """
        try:
            best_algorithm_name = self._select_best_clustering_algorithm(data)
            X_scaled = self._prepare_clustering_features(data)

            if X_scaled.shape[0] < 2:
                return {'error': 'Not enough data points to perform clustering.'}

            if best_algorithm_name == "K-Means":
                n_clusters = min(5, max(2, X_scaled.shape[0] // 10))
                algorithm = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            else:  # DBSCAN
                algorithm = DBSCAN(eps=0.75, min_samples=max(2, X_scaled.shape[0] // 20))

            labels = algorithm.fit_predict(X_scaled)

            score = 0
            # Silhouette score is only valid for 2 or more clusters
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)

            return {
                'algorithm': best_algorithm_name,
                'n_clusters': len(set(labels)),
                'silhouette_score': score,
                'labels': labels.tolist()
            }
        except Exception as e:
            return {'error': f"Clustering failed: {e}"}

    def _select_best_clustering_algorithm(self, data):
        """
        Selects the best clustering algorithm based on heuristics and evaluation.

        - Small datasets (<100): Prefers K-Means for simplicity.
        - Medium datasets (<1000): Evaluates both K-Means and DBSCAN.
        - Large datasets (>=1000): Prefers DBSCAN for its ability to handle noise.

        Args:
            data (list or pd.DataFrame): The dataset to be clustered.

        Returns:
            str: The name of the recommended clustering algorithm ('K-Means' or 'DBSCAN').
        """
        n_samples = len(data)
        if n_samples < 100:
            return "K-Means"

        if n_samples < 1000:
            kmeans_score = self._evaluate_kmeans(data)
            dbscan_score = self._evaluate_dbscan(data)
            return "K-Means" if kmeans_score >= dbscan_score else "DBSCAN"

        return "DBSCAN"

    def _evaluate_kmeans(self, data):
        """Evaluates K-Means performance using the silhouette score."""
        try:
            X_scaled = self._prepare_clustering_features(data)
            if X_scaled.shape[0] < 2:
                return 0

            n_clusters = min(5, max(2, X_scaled.shape[0] // 10))
            if n_clusters >= X_scaled.shape[0]: return 0

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X_scaled)

            return silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
        except:
            return 0  # Return low score on any error

    def _evaluate_dbscan(self, data):
        """Evaluates DBSCAN performance using the silhouette score."""
        try:
            X_scaled = self._prepare_clustering_features(data)
            if X_scaled.shape[0] < 2:
                return 0

            dbscan = DBSCAN(eps=0.75, min_samples=max(2, X_scaled.shape[0] // 20))
            labels = dbscan.fit_predict(X_scaled)

            # Ignore noise points for scoring if they exist
            if -1 in labels:
                core_samples_mask = labels != -1
                if np.sum(core_samples_mask) < 2 or len(set(labels[core_samples_mask])) < 2:
                    return 0  # Not enough clusters to score
                return silhouette_score(X_scaled[core_samples_mask], labels[core_samples_mask])

            return silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
        except:
            return 0  # Return low score on any error

    # 4. Private Helper Methods

    def _get_skill_feature_list(self):
        """Returns a sorted list of skills to ensure consistent feature order."""
        return sorted(self.skill_weights.keys())

    def _vectorize_candidate(self, candidate_record, skill_feature_list):
        """
        Converts a single candidate data record into a numerical feature vector.

        Args:
            candidate_record (pd.Series or dict): The candidate's data.
            skill_feature_list (list): An ordered list of skills for one-hot encoding.

        Returns:
            list: The numerical feature vector.
        """
        skills = candidate_record.get('skills', [])

        # Base features
        feature_vector = [
            candidate_record.get('similarity_score', 0),
            candidate_record.get('experience_years', 0),
            len(skills),
            1 if any(skill in self.skill_weights for skill in skills) else 0
        ]

        # One-hot encoded skill features
        skill_features = [1 if skill in skills else 0 for skill in skill_feature_list]
        feature_vector.extend(skill_features)

        return feature_vector

    def _prepare_clustering_features(self, data):
        """Extracts and scales a standard set of features for clustering tasks."""
        df = pd.DataFrame(data) if isinstance(data, list) else data
        if df.empty:
            return np.array([])

        features = df.apply(lambda row: [
            row.get('similarity_score', 0),
            row.get('experience_years', 0),
            len(row.get('skills', [])),
            1 if any(skill in self.skill_weights for skill in row.get('skills', [])) else 0
        ], axis=1).tolist()

        X = np.array(features)
        return StandardScaler().fit_transform(X)

    def _generate_recommendations(self):
        """Generates actionable hiring recommendations based on the analysis."""
        recommendations = []
        if not self.pattern_analysis:
            return recommendations

        # Recommendation 1: Focus on top skills
        if 'skill_analysis' in self.pattern_analysis:
            top_skills = [s[0] for s in self.pattern_analysis['skill_analysis']['top_successful_skills'][:5]]
            recommendations.append({
                'title': 'Prioritize High-Success Skills',
                'details': f"Candidates with skills like {', '.join(top_skills)} have a higher hiring rate. Focus sourcing and screening efforts on these."
            })

        # Recommendation 2: Target ideal experience level
        if 'experience_analysis' in self.pattern_analysis:
            exp_hired = self.pattern_analysis['experience_analysis']['avg_experience_hired']
            recommendations.append({
                'title': 'Target Optimal Experience Level',
                'details': f"Hired candidates have an average of {exp_hired:.1f} years of experience. Consider this a benchmark for ideal candidates."
            })

        # Recommendation 3: Use similarity score as a filter
        if 'similarity_analysis' in self.pattern_analysis:
            sim_hired = self.pattern_analysis['similarity_analysis']['avg_similarity_hired']
            recommendations.append({
                'title': 'Leverage Similarity Scores',
                'details': f"Successful hires score around {sim_hired:.2f} on average. Use scores above this threshold as a strong positive signal."
            })

        return recommendations



if __name__ == "__main__":
    # # --- 1. Setup: Create a richer sample dataset ---
    # sample_data = [
    #     {'candidate_id': 1, 'name': 'Ahmed Ali', 'similarity_score': 0.85, 'experience_years': 5,
    #      'skills': ['Python', 'Django', 'API'], 'hired': True},
    #     {'candidate_id': 2, 'name': 'Sara Mohamed', 'similarity_score': 0.72, 'experience_years': 3,
    #      'skills': ['JavaScript', 'React', 'Node.js'], 'hired': False},
    #     {'candidate_id': 3, 'name': 'John Doe', 'similarity_score': 0.91, 'experience_years': 8,
    #      'skills': ['Python', 'Machine Learning', 'AWS'], 'hired': True},
    #     {'candidate_id': 4, 'name': 'Jane Smith', 'similarity_score': 0.65, 'experience_years': 4,
    #      'skills': ['JavaScript', 'Vue.js', 'CSS'], 'hired': False},
    #     {'candidate_id': 5, 'name': 'Li Wei', 'similarity_score': 0.88, 'experience_years': 6,
    #      'skills': ['Python', 'Flask', 'Docker', 'API'], 'hired': True},
    #     {'candidate_id': 6, 'name': 'Fatima Khan', 'similarity_score': 0.78, 'experience_years': 3,
    #      'skills': ['React', 'Node.js', 'TypeScript'], 'hired': True},
    #     {'candidate_id': 7, 'name': 'Carlos Rossi', 'similarity_score': 0.59, 'experience_years': 2,
    #      'skills': ['Python', 'Django'], 'hired': False},
    #     {'candidate_id': 8, 'name': 'Maria Garcia', 'similarity_score': 0.82, 'experience_years': 7,
    #      'skills': ['Machine Learning', 'TensorFlow', 'Python'], 'hired': True},
    #     {'candidate_id': 9, 'name': 'Kenji Tanaka', 'similarity_score': 0.71, 'experience_years': 5,
    #      'skills': ['JavaScript', 'AWS', 'Node.js'], 'hired': False},
    #     {'candidate_id': 10, 'name': 'Aisha Bello', 'similarity_score': 0.93, 'experience_years': 9,
    #      'skills': ['Python', 'AWS', 'Docker', 'API'], 'hired': True},
    #     {'candidate_id': 11, 'name': 'David Chen', 'similarity_score': 0.75, 'experience_years': 4,
    #      'skills': ['React', 'JavaScript', 'GraphQL'], 'hired': False},
    #     {'candidate_id': 12, 'name': 'Isabella Costa', 'similarity_score': 0.80, 'experience_years': 6,
    #      'skills': ['Python', 'API', 'Flask'], 'hired': True}
    # ]
    #
    # with open('hiring_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(sample_data, f, indent=2)

    #  2. Initialize and run the full analytics pipeline
    analytics = AdvancedAnalytics()

    if analytics.load_hiring_data():
        print("\n🔍 Running hiring pattern analysis...")
        analytics.analyze_hiring_patterns()

        print("⚖️ Calculating skill weights...")
        analytics.calculate_skill_weights()

        print("🤖 Building hiring prediction model...")
        model_performance = analytics.build_hiring_prediction_model()
        if "error" not in model_performance:
            print(f"🏆 Best Model Selected: {model_performance['best_model']}")

            #  3. Generate and display report
            print("\n📄 Generating final performance report...")
            report = analytics.generate_performance_report()
            print(json.dumps(report, indent=2))

            #  4. Predict for a new candidate
            print("\n✨ Predicting for a new candidate...")
            new_candidate = {
                'similarity_score': 0.89,
                'experience_years': 7,
                'skills': ['Python', 'AWS', 'Machine Learning']
            }
            prediction_result = analytics.predict_hiring_success(new_candidate)
            print("Prediction Result:", json.dumps(prediction_result, indent=2))

            #  5. Create and save visualizations
            print("\n📊 Creating visualizations...")
            viz_result = analytics.create_visualizations()
            print(viz_result['message'])

            #  6. Perform Clustering
            print("\n🌀 Performing candidate clustering...")
            clustering_result = analytics.get_optimal_clustering(analytics.hiring_data)
            print("Clustering Result:", json.dumps(clustering_result, indent=2))
