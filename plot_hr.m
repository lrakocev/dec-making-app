%% ingest + format

appr_avoid_hr = read_hr_csv_to_table("approach_avoid_hr_0304.xlsx");
social_hr = read_hr_csv_to_table("social_hr_0304.xlsx");
moral_hr = read_hr_csv_to_table("moral_hr_0304.xlsx");
prob_hr = read_hr_csv_to_table("probability_hr_0304.xlsx");

%% bar plots across tasks

save_to = 'C:\Users\lrako\OneDrive\Documents\dec-making-app\final_run_hr\';
tasks = {"appr avoid", "social", "moral", "prob"};
tables = {appr_avoid_hr, social_hr, moral_hr, probability_hr};
bar_type = "percent min hr";

bar_plots_by_trial(tasks,tables,bar_type,save_to,1)
bar_plots_by_subject(tasks,tables,bar_type,save_to,1)

%% 3D plot for separation

save_to =  'C:\Users\lrako\OneDrive\Documents\dec-making-app\final_run_hr\';
tasks = {"appr avoid", "social", "moral", "prob"};
tables = {appr_avoid_hr, social_hr, moral_hr, probability_hr};
colors = ["r", "k", "c", "m"];

separation_across_tasks_by_subject_3d_hr(tasks, tables, colors,save_to)
separation_across_tasks_by_trial_3d_hr(tasks, tables, colors,save_to)

%% timing plots within task

tasks = {"appr avoid", "social", "moral", "prob"};
tables = {appr_avoid_hr, social_hr, moral_hr, probability_hr};
save_to =  'C:\Users\lrako\OneDrive\Documents\dec-making-app\final_run_hr\';

for i = 1:length(tasks)
    tit = tasks(i);
    table = tables(i);
    bar_plots_within_task_by_trial_hr(table{1}, tit, "cost", save_to)
    bar_plots_within_task_by_trial_hr(table{1}, tit, "reward", save_to)

    bar_plots_within_task_by_subject_hr(table{1}, tit, "cost", save_to)
    bar_plots_within_task_by_subject_hr(table{1}, tit, "reward", save_to)
end

