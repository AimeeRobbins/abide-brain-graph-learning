if __name__ == '__main__':
    # Import packages
    from pathlib import Path
    from abide_data_loader import AbideDataLoader
    import pandas as pd
    import statistics  

    directory = Path("../abide_data/Outputs/cpac/filt_global/rois_cc200")
    pheno_df = pd.read_csv('../Phenotypic_V1_0b_preprocessed1.csv')
    data_dir = '../abide_data/Outputs/cpac/filt_global/rois_cc200'
    data_loader = AbideDataLoader(data_dir=data_dir, pheno_df=pheno_df)

    subjects = []

    for file in directory.iterdir():
        print(file.name)
        # Load a single subject
        _, y, site, sid, sex, age = data_loader.load_subject(file.name)
        subjects.append({
            "sid": sid,
            "y": y,
            "site": site,
            "sex": sex,
            "age": age
        })


    for site in sorted({subject["site"] for subject in subjects}):
        asd_count = sum(
            subject["y"] == 1
            for subject in subjects
            if subject["site"] == site
        )

        control_count = sum(
            subject["y"] == 0
            for subject in subjects
            if subject["site"] == site
        )

        total = asd_count + control_count

        print(f"{site}: ASD = {asd_count}, Control = {control_count}, Total = {total}")

