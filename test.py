from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from get_data import txt_to_pd_WISDM
import tsfresh
# timeseries, y = load_robot_execution_failures()

# print(timeseries, y)

# extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
# print(extracted_features)
# impute(extracted_features)
# features_filtered = select_features(extracted_features, y)

# print(features_filtered)

# features_filtered_direct = extract_relevant_features(timeseries, y,
#                                                      column_id='id', column_sort='time')

# print(features_filtered_direct)
data = txt_to_pd_WISDM()
data = data[0:1000]
print(type(data['x-axis'][0]))

print(data)
y = data['activity']
data = data.drop(['activity'], axis=1)
ts = tsfresh.extract_features(data, column_id='user_id', 
    column_sort ='timestamp', column_kind=None, column_value=None)

print(ts)