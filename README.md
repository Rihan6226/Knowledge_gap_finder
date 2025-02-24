# Personalized Student Recommendations

## Project Overview

This project analyzes a student's quiz performance and provides personalized recommendations to help them improve their preparation for NEET exams. By evaluating their historical quiz data and recent quiz submissions, the solution identifies weak areas, trends in performance, and gaps in knowledge to generate actionable recommendations.

## Features

- **Analyze Quiz Data**: Review the student's latest quiz performance, including their responses and accuracy.
- **Generate Recommendations**: Based on performance gaps, the system suggests topics to focus on and additional questions for practice.
- **Topic-Based Suggestions**: Leveraging PDF content from relevant chapters, the system generates a personalized learning plan.
- **Bonus**: The system provides insights into the student's strengths and weaknesses through creative labels.

## Solution Flow

1. **Data Exploration**: 
   - Historical Quiz Data (last 5 quizzes for each user) and Current Quiz Data (latest submission).
   
2. **Gap Identification**: 
   - Based on the user's incorrect responses, the system identifies areas needing improvement.
   
3. **Generate Recommendations**: 
   - The system retrieves relevant chapter content, processes it, and generates questions to strengthen weak areas.
   
4. **User Insights**: 
   - Provides feedback on strengths and weaknesses with a creative persona-based approach.

## Setup Instructions

### Requirements

- Python 3.x
- Required Python Libraries:
  - `sentence-transformers`
  - `torch`
  - `sklearn`
  - `pdfplumber`
  - `google-generativeai`
  - `python-dotenv`

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/personalized-student-recommendations.git
   cd personalized-student-recommendations
   ```
2. Install the dependencies:

 ```bash
   pip install -r requirements.txt
```

3. Set up environment variables:

Create a .env file in the root of the project with the following content:
```bash
   API_KEY=your_api_key_here
```
4. Place the necessary JSON files for Quiz data and Response data and PDF files (LLQT.json, rJvd7g.json, etc.) in the root directory or the specified folders.

## Running the Application

1. Run the main script:
   ```bash
   python main.py
   ```
## The system will process the data and print the recommendations.

## Code Explanation

### `main.py`

The main script orchestrates the execution flow by loading the quiz data, processing responses, and invoking the `gap_finder` and `pretty_print` functions to generate and display the recommendations.

### `utils.py`

This file contains utility functions for:
- **Processing User Responses**: Analyzing and comparing user responses with correct answers.
- **Text Extraction from PDFs**: Extracting chapter content from PDF files.
- **Text Chunking and Embedding Generation**: Breaking down extracted content into chunks, generating embeddings using a pre-trained model, and calculating cosine similarities.
- **Generative AI for Titles and Questions**: Using the Google Generative AI model to generate topics and questions based on the textbook content.
- **Gap Finder**: Identifying performance gaps and linking them with relevant chapter content.

## Example Output

```plaintext
User needs to improve in these topics:

1. Human Digestive System
2. Nervous System in Frogs

Try these questions after diving deep into the suggested topics:

Human Digestive System:
1. What is the role of the liver in digestion?
    a. Secretion of bile
    b. Storage of glycogen
    c. Digestion of fats
    d. Absorption of nutrients
    Correct Answer: Secretion of bile

Nervous System in Frogs:
1. Which of the following is true about the frog's nervous system?
    a. It has a brain but no spinal cord
    b. It has ten pairs of cranial nerves
    c. It lacks an autonomic nervous system
    d. It has a single optic lobe
    Correct Answer: It has ten pairs of cranial nerves
```
## Future Improvements

- **Real-Time Data Processing**: Implement a real-time quiz submission handler.
- **More Advanced AI Models**: Explore the use of larger language models for more personalized and contextual recommendations.
- **User Feedback System**: Allow users to provide feedback on the recommendations to continuously improve the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
