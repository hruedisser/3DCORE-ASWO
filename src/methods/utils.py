
def extract_row(row):
    
    params = [
        row['Longitude'],
        row['Latitude'],
        row['Inclination'],
        row['Diameter 1 AU'],
        row['Aspect Ratio'],
        row['Launch Radius'],
        row['Launch Velocity'],
        row['Expansion Rate'],
        row['Background Drag'],
        row['Background Velocity'],
        row['T_Factor'],
        row['Magnetic Decay Rate'],
        row['Magnetic Field Strength 1 AU'],        
    ]

    return params

def get_iparams_live(*modelstatevars):
    
    model_kwargs = {
        "ensemble_size": int(1), #2**17
        "iparams": {
            "cme_longitude": {
                "distribution": "fixed",
                "default_value": modelstatevars[0]
            },
            "cme_latitude": {
                "distribution": "fixed",
                "default_value": modelstatevars[1]
            },
            "cme_inclination": {
                "distribution": "fixed",
                "default_value": modelstatevars[2]
            },
            "cme_diameter_1au": {
                "distribution": "fixed",
                "default_value": modelstatevars[3]
            },
            "cme_aspect_ratio": {
                "distribution": "fixed",
                "default_value": modelstatevars[4]
            },
            "cme_launch_radius": {
                "distribution": "fixed",
                "default_value": modelstatevars[5]
            },
            "cme_launch_velocity": {
                "distribution": "fixed",
                "default_value": modelstatevars[6]
            },
            "t_factor": {
                "distribution": "fixed",
                "default_value": modelstatevars[10]
            },
            "cme_expansion_rate": {
                "distribution": "fixed",
                "default_value": modelstatevars[7]
            },
            "magnetic_decay_rate": {
                "distribution": "fixed",
                "default_value": modelstatevars[11]
            },
            "magnetic_field_strength_1au": {
                "distribution": "fixed",
                "default_value": modelstatevars[12]
            },
            "background_drag": {
                "distribution": "fixed",
                "default_value": modelstatevars[8]
            },
            "background_velocity": {
                "distribution": "fixed",
                "default_value": modelstatevars[9]
            }
        }
    }
    
    return model_kwargs


def get_modelkwargs_ranges(fittingstate_values):


    ensemble_size = fittingstate_values[0]
        
    model_kwargs = {
        "ensemble_size": ensemble_size, #2**17
        "iparams": {
            "cme_longitude": {
                "maximum": fittingstate_values[1][1],
                "minimum": fittingstate_values[1][0]
            },
            "cme_latitude": {
                "maximum": fittingstate_values[2][1],
                "minimum": fittingstate_values[2][0]
            },
            "cme_inclination": {
                "distribution": "uniform",
                "maximum": fittingstate_values[3][1],
                "minimum": fittingstate_values[3][0]
            },
            "cme_diameter_1au": {
                "maximum": fittingstate_values[4][1],
                "minimum": fittingstate_values[4][0]
            },
            "cme_aspect_ratio": {
                "maximum": fittingstate_values[5][1],
                "minimum": fittingstate_values[5][0]
            },
            "cme_launch_radius": {
                "distribution": "uniform",
                "maximum": fittingstate_values[6][1],
                "minimum": fittingstate_values[6][0]
            },
            "cme_launch_velocity": {
                "maximum": fittingstate_values[7][1],
                "minimum": fittingstate_values[7][0]
            },
            "t_factor": {
                "maximum": fittingstate_values[11][1],
                "minimum": fittingstate_values[11][0],
            },
            "cme_expansion_rate": {
                "distribution": "uniform",
                "maximum": fittingstate_values[8][1],
                "minimum": fittingstate_values[8][0],
            },
            "magnetic_decay_rate": {
                "distribution": "uniform",
                "maximum": fittingstate_values[12][1],
                "minimum": fittingstate_values[12][0],
            },
            "magnetic_field_strength_1au": {
                "maximum": fittingstate_values[13][1],
                "minimum": fittingstate_values[13][0],
            },
            "background_drag": {
                "distribution": "uniform",
                "maximum": fittingstate_values[9][1],
                "minimum": fittingstate_values[9][0],
            },
            "background_velocity": {
                "distribution": "uniform",
                "maximum": fittingstate_values[10][1],
                "minimum": fittingstate_values[10][0],
            }
        }
    }
    
    for param, values in model_kwargs["iparams"].items():
        if values["maximum"] == values["minimum"]:
            values["distribution"] = "fixed"
            values["default_value"] = values["minimum"]
            del values["maximum"]
            del values["minimum"]
    
    return model_kwargs