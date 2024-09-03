# How to run the analysis

1. **Create Conda Environment:**
   Create the conda environment using the provided `environment.yml` file:
   ```sh
   conda env create -f environment.yml
   ```

2. **Create Required Directories:**
   Create the necessary directories for data and model checkpoints:
   ```sh
   mkdir deliverables/data
   mkdir deliverables/tic_checkpoints
   ```

3. **Place Files:**
   - Put your `.txt` files in the `data` directory.
   - Ensure your model checkpoints are in the `tic_checkpoints` directory.

4. **Setup Environment Variables:**
   Create a `.env` file in the root directory and add the following content:
   ```env
   TRENDSCANNER_ENTITY_EXTRACTION_KEY=your_key_here
   CHECKPOINT='2024-07-31 00:05:15-distilbert-base-uncased'
   SEED='seed_1'
   ```
   Note that the checkpoint and seed from above are just examples. Replace the values with the once you actually plan on using. 

5. **Activate Conda Environment:**
   Activate the conda environment:
   ```sh
   conda activate report_env
   ```

6. **Run the Script:**
   Execute the script:
   ```sh
   python deliverables/report.py
   ```