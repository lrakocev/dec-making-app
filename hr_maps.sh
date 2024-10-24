#!/bin/sh -x
while read -r id; do
	while read -r story; do
		echo "$id"
		echo "$story"
		python heart_rate_analysis.py $id $story
done < <(python grab_stories_for_eyetracking.py $id)
done < <(python grab_ids.py)