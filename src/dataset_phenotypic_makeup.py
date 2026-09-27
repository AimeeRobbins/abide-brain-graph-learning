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

    ages = []
    ages_asd = []
    ages_control = []
    male_asd = 0
    female_asd = 0
    male_control = 0
    female_control = 0

    for file in directory.iterdir():
        print(file.name)
        # Load a single subject
        _, y, site, sid, sex, age = data_loader.load_subject(file.name)

        if(y == 1): #asd
            ages_asd.append(age)
            if(sex == 1): # male
                male_asd += 1
            else: # female
                female_asd += 1
        else: # control
            ages_control.append(age)
            if(sex == 1):
                male_control += 1
            else:
                female_control += 1

    # Calculate mean and standard deviation
    ages = ages_asd + ages_control
    mean_age = sum(ages) / len(ages)
    std_age = statistics.stdev(ages)

    mean_age_asd = sum(ages_asd) / len(ages_asd)
    std_age_asd = statistics.stdev(ages_asd)
    mean_age_control = sum(ages_control) / len(ages_control)
    std_age_control = statistics.stdev(ages_control)
    total_asd = male_asd + female_asd
    total_control = male_control + female_control
    total = total_asd + total_control

    print(f"mean_age = {mean_age} ")
    print(f"std_age = {std_age} ")
    print(f"mean_age_asd = {mean_age_asd} ")
    print(f"std_age_asd = {std_age_asd} ")
    print(f"mean_age_control = {mean_age_control} ")
    print(f"std_age_control = {std_age_control} ")
    print(f"total = {total}")
    print(f"total_asd = {total_asd} ")
    print(f"male_asd = {male_asd}, ({male_asd*100/total_asd})%")
    print(f"female_asd = {female_asd}, ({female_asd*100/total_asd})%")
    print(f"total_control = {total_control} ")
    print(f"male_control = {male_control}, ({male_control*100/total_control})%")
    print(f"female_control = {female_control}, ({female_control*100/total_control})% ")

    