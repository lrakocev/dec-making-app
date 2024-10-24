%% ingest + format

appr_avoid_eye = read_eye_csv_to_table("approach_avoid_eyetracking_0304.xlsx");
moral_eye = read_eye_csv_to_table("moral_eyetracking_0304.xlsx");
social_eye = read_eye_csv_to_table("social_eyetracking_0304.xlsx");
prob_eye = read_eye_csv_to_table("probability_eyetracking_0304.xlsx");

%% linear plots w/ stats
save_to = 'C:\Users\lrako\OneDrive\Documents\dec-making-app\final_run\';
plot_timing_vs_eng_inc(moral_eye, 'moral', save_to)
plot_timing_vs_eng_inc(social_eye, 'social',save_to)
plot_timing_vs_eng_inc(prob_eye, 'prob',save_to)
plot_timing_vs_eng_inc(appr_avoid_eye, 'appr_avoid',save_to)

%% dec making plots

map_type = "decision_timing";
tit = "appr_avoid";
table = appr_avoid_eye;

if isequal(map_type, "decision_timing")
    table.decision_timing = get_arr(table.decision_timing);
    table = table(table.decision_timing ~= 0, :);
elseif isequal(map_type, "num_local_maxima")
    table.num_maxes = get_arr(table.num_maxes);
elseif isequal(map_type, "num_inc_in_eng")
    table.big_ups = get_arr(table.big_ups);
end

table.real_r = get_arr(table.real_r);
table.real_c = get_arr(table.real_c);

make_dec_timing_map(table, tit, map_type)

%% bar plots across tasks

save_to = 'C:\Users\lrako\OneDrive\Documents\dec-making-app\final_run\';
tasks = {"appr avoid", "social", "moral", "prob"};
tables = {appr_avoid_eye, social_eye, moral_eye, prob_eye};
bar_type = "num inc in eng";

bar_plots_by_trial(tasks,tables,bar_type,save_to,0)
bar_plots_by_subject(tasks,tables,bar_type,save_to,0)

%% 2D plot for separation

save_to =  'C:\Users\lrako\OneDrive\Documents\dec-making-app\final_run\';
tasks = {"appr avoid", "social", "moral", "prob"};
tables = {appr_avoid_eye, social_eye, moral_eye, prob_eye};
colors = ["r", "k", "c", "m"];

separation_across_tasks_by_subject(tasks, tables, colors,save_to)
separation_across_tasks_by_trial(tasks, tables, colors,save_to)

%% 3D plot for separation

save_to =  'C:\Users\lrako\OneDrive\Documents\dec-making-app\final_run\';
tasks = {"appr avoid", "social", "moral", "prob"};
tables = {appr_avoid_eye, social_eye, moral_eye, prob_eye};
colors = ["r", "k", "c", "m"];

separation_across_tasks_by_subject_3d(tasks, tables, colors,save_to)
separation_across_tasks_by_trial_3d(tasks, tables, colors,save_to)

%% timing plots within task

tasks = {"appr avoid", "social", "moral", "prob"};
tables = {appr_avoid_eye, social_eye, moral_eye, prob_eye};
save_to =  'C:\Users\lrako\OneDrive\Documents\dec-making-app\final_run\';

for i = 1:length(tasks)
    tit = tasks(i);
    table = tables(i);
    bar_plots_within_task_by_trial(table{1}, tit, "cost",save_to)
    bar_plots_within_task_by_trial(table{1}, tit, "reward",save_to)

    bar_plots_within_task_by_subject(table{1}, tit, "cost",save_to)
    bar_plots_within_task_by_subject(table{1}, tit, "reward",save_to)
end