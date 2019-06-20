import solarwind_analysis as swa

if __name__=="__main__":
    swa.import_solar_wind_plasma_data()
    swa.HI_processing_schematic()  # Makes figure 1
    swa.spacecraft_orbits()  # Makes figure 2
    swa.insitu_time_series_acf()  # Makes figure 3
    swa.HI_time_series_acf()  # Makes figure 4
    swa.rolling_acf()  # Makes figure 5.
    swa.hi1a_thomson_sphere_and_parker_spiral()  # Makes figure 6
    swa.lagged_correlation()  # Makes figure 7 and 8
