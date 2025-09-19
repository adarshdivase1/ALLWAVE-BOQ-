# AI-Powered BOQ Generator for AV Solutions

An intelligent Bills of Quantities generator specifically designed for Audio-Visual solution companies, featuring AI-powered product recommendations and AVIXA compliance checking.

## Features

- 🤖 **AI-Powered Recommendations**: Smart product selection based on project requirements
- 📋 **AVIXA Compliance**: Automatic compliance checking with industry standards
- 📊 **Professional Reports**: Generate detailed BOQ reports in PDF format
- 🔍 **Smart Product Matching**: Advanced search and matching algorithms
- 📈 **Cost Optimization**: Budget-aware product recommendations
- 🏢 **Multi-Company Support**: Process data from multiple vendor catalogs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/boq-generator.git
cd boq-generator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data models reports temp
```

## Usage

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Data Processing
- Upload your Excel file containing product catalogs from multiple companies
- The system will process all sheets and extract product information
- Cleaned data will be saved to `data/processed_products.json`

### 3. Generate BOQ
- Fill in project details (name, client, type, requirements)
- Select specific requirements from the available options
- Click "Generate BOQ" to create an AI-powered recommendation
- Download the generated report as PDF

### 4. Model Training
- Configure training parameters
- Train custom models for better recommendations
- Monitor model performance metrics

## Project Structure

```
boq-generator/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── data/                 # Data storage
│   ├── processed_products.json
│   ├── avixa_guidelines.json
│   └── sample_boqs.json
├── models/              # ML models
│   ├── boq_model.py
│   ├── data_processor.py
│   └── model_trainer.py
├── utils/               # Utility functions
│   ├── excel_processor.py
│   ├── boq_generator.py
│   └── avixa_compliance.py
├── components/          # UI components
│   ├── ui_components.py
│   └── report_generator.py
└── notebooks/           # Jupyter notebooks
    ├── data_exploration.ipynb
    └── model_training.ipynb
```

## Configuration

### Environment Variables
Create a `.env` file with the following variables:
```
HF_API_TOKEN=your_huggingface_token
STREAMLIT_SERVER_PORT=8501
```

### AVIXA Standards
The system includes built-in AVIXA compliance checking for:
- Display distance ratios
- Audio coverage requirements
- Control system redundancy
- Cable management standards
- Room acoustics guidelines

## Excel File Format

Your Excel file should contain multiple sheets (one per company) with the following columns:
- `model_no`: Product model number
- `description`: Product description
- `price`: Product price (optional)
- `specifications`: Technical specifications (optional)
- Additional columns will be processed automatically

## API Integration

### Hugging Face Models
The system uses Hugging Face transformers for:
- Product embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Text similarity matching
- Feature extraction

### Custom Models
Train custom models for:
- Product recommendation
- Price prediction
- Compatibility checking
- Compliance validation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Requirements

- Python 3.8+
- Streamlit 1.28+
- pandas, numpy, scikit-learn
- transformers, torch
- openpyxl for Excel processing
- plotly for visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Contact: support@yourcompany.com
- Documentation: [Link to detailed docs]

## Roadmap

- [ ] Integration with more ML models
- [ ] Real-time pricing updates
- [ ] Advanced visualization dashboard
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Integration with ERP systems

## Acknowledgments

- AVIXA for industry standards
- Hugging Face for ML models
- Streamlit community for the framework
- Contributors and beta testers
