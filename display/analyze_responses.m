
%%
clear all;
KbName('UnifyKeyNames');

ONE = KbName('1!');
TWO = KbName('2@');
THREE = KbName('3#');
FOUR = KbName('4$');
FIVE = KbName('5%');
all_responses = [ONE TWO THREE FOUR FIVE];

UP_key = KbName('UpArrow');
DOWN_key = KbName('DownArrow');
CONTEXT_QUESTIONS = [5 9 27 28 29 30 34 35 36 37 38 39];
% I'll score context questions as cheating category = 1, paranoid category
% = 2
CONTEXT_ANSWERS = {};
QUESTIONS = {};
POSSIBLE_ANSWERS = {};
QUESTIONS{1} = 'What was the girl doing when the phone rang?';
POSSIBLE_ANSWERS{1,1} = 'Lying on the bed';
POSSIBLE_ANSWERS{1,2} = 'Sitting in a chair';
CORRECT_ANSWER(1) = 1;
QUESTIONS{2} = 'What was the gray-haired man''s name?';
POSSIBLE_ANSWERS{2,1} = 'Lee';
POSSIBLE_ANSWERS{2,2} = 'Louis';
CORRECT_ANSWER(2) = 1;
QUESTIONS{3} = 'What was the second man''s name?';
POSSIBLE_ANSWERS{3,1} = 'Arthur';
POSSIBLE_ANSWERS{3,2} = 'Alan';
CORRECT_ANSWER(3) = 1;
QUESTIONS{4} = 'What was Arthur''s wife name?';
POSSIBLE_ANSWERS{4,1} = 'Joanie';
POSSIBLE_ANSWERS{4,2} = 'Jenny';
CORRECT_ANSWER(4) = 1;
QUESTIONS{5} = 'What was Lee''s girlfriend name?';
POSSIBLE_ANSWERS{5,1} = 'Joanie';
POSSIBLE_ANSWERS{5,2} = 'Rosie'; % change to don't know???
CORRECT_ANSWER(5) = 0;
CONTEXT_ANSWERS{5,1} = 1; % cheating response
CONTEXT_ANSWERS{5,2} = 2; % paranoid response
QUESTIONS{6} = 'Why did Arthur call Lee?';
POSSIBLE_ANSWERS{6,1} = 'Because Joanie didn''t come home, and he wanted to know if Lee noticed when she left';
POSSIBLE_ANSWERS{6,2} = 'Because Joanie didn''t come home, and he wanted to know if she was at Lee''s house';
CORRECT_ANSWER(6) = 1;
QUESTIONS{7} = 'Why was Arthur worried?';
POSSIBLE_ANSWERS{7,1} = 'Because he thought Joanie was cheating on him';
POSSIBLE_ANSWERS{7,2} = 'Because he thought Joanie was drunk';
CORRECT_ANSWER(7) = 1;
QUESTIONS{8} = 'How did Lee try to calm Arthur regarding Joanie?';
POSSIBLE_ANSWERS{8,1} = 'He said that Joanie probably went with the Ellenbogens to another place';
POSSIBLE_ANSWERS{8,2} = 'He said that Joanie probably didn''t find a cab and waited for a ride back home';
CORRECT_ANSWER(8) = 1;
QUESTIONS{9} = 'What did you think of Arthur when he said, "I have a feeling she went to work on some bastard in the kitchen."?';
POSSIBLE_ANSWERS{9,1} = 'He knew what he was talking about';
POSSIBLE_ANSWERS{9,2} = 'He was being paranoid';
CORRECT_ANSWER(9) = 0;
CONTEXT_ANSWERS{9,1} = 1;
CONTEXT_ANSWERS{9,2} = 2;
QUESTIONS{10} = 'How many years were Arthur and Joanie together?';
POSSIBLE_ANSWERS{10,1} = 'Five';
POSSIBLE_ANSWERS{10,2} = 'Fifteen';
CORRECT_ANSWER(10) = 1;
QUESTIONS{11} = 'What did Lee advise Arthur to do?';
POSSIBLE_ANSWERS{11,1} = 'To get in bed and relax';
POSSIBLE_ANSWERS{11,2} = 'To keep calling the Ellenbogens';
CORRECT_ANSWER(11) = 1;
QUESTIONS{12} = 'How did Arthur feel about himself throughout the conversation?';
POSSIBLE_ANSWERS{12,1} = 'That he was losing his mind';
POSSIBLE_ANSWERS{12,2} = 'That he was very sophisticated';
CORRECT_ANSWER(12) = 1;
QUESTIONS{13} = 'What did Arthur expect every night when he came home?';
POSSIBLE_ANSWERS{13,1} = 'To find his wife cheating on him';
POSSIBLE_ANSWERS{13,2} = 'To find his wife waiting for him';
CORRECT_ANSWER(13) = 1;
QUESTIONS{14} = 'What was the girl doing throughout the conversation?';
POSSIBLE_ANSWERS{14,1} = 'Listening while handling the cigarettes';
POSSIBLE_ANSWERS{14,2} = 'Listening while reading a book';
CORRECT_ANSWER(14) = 1;
QUESTIONS{15} = 'What did Lee think was Arthur''s problem?';
POSSIBLE_ANSWERS{15,1} = 'That he was torturing himself and being paranoid';
POSSIBLE_ANSWERS{15,2} = 'That he was drinking too much and losing his judgment';
CORRECT_ANSWER(15) = 1;
QUESTIONS{16} = 'What did Lee think about Joanie?';
POSSIBLE_ANSWERS{16,1} = 'That she was wonderful and smart';
POSSIBLE_ANSWERS{16,2} = 'That she was beautiful and superficial';
CORRECT_ANSWER(16) = 1;
QUESTIONS{17} = 'Did Arthur think Joanie was smart?';
POSSIBLE_ANSWERS{17,1} = 'No';
POSSIBLE_ANSWERS{17,2} = 'Yes';
CORRECT_ANSWER(17) = 1;
QUESTIONS{18} = 'What did Joanie buy Arthur once?';
POSSIBLE_ANSWERS{18,1} = 'A suit';
POSSIBLE_ANSWERS{18,2} = 'A watch';
CORRECT_ANSWER(18) = 1;
QUESTIONS{19} = 'What did Arthur ask Lee towards the end of the conversation?';
POSSIBLE_ANSWERS{19,1} = 'If he could come over to Lee''s place';
POSSIBLE_ANSWERS{19,2} = 'If Lee could come over to Arthur''s home';
CORRECT_ANSWER(19) = 1;
QUESTIONS{20} = 'What did Lee answer?';
POSSIBLE_ANSWERS{20,1} = 'That it was better for Arthur to be there, when Joanie comes back home';
POSSIBLE_ANSWERS{20,2} = 'That he was tired, and rather go to sleep';
CORRECT_ANSWER(20) = 1;
QUESTIONS{21} = 'Did Lee want Arthur to come over?';
POSSIBLE_ANSWERS{21,1} = 'No';
POSSIBLE_ANSWERS{21,2} = 'Yes';
CORRECT_ANSWER(21) = 1;
QUESTIONS{22} = 'What did the girl ask after Lee hung up?';
POSSIBLE_ANSWERS{22,1} = '"What did he say?"';
POSSIBLE_ANSWERS{22,2} = '"What''s wrong with him?"';
CORRECT_ANSWER(22) = 1;
QUESTIONS{23} = 'What did the girl think about how Lee handled the situation?';
POSSIBLE_ANSWERS{23,1} = 'That he handled it very well';
POSSIBLE_ANSWERS{23,2} = 'That he should have been more assertive';
CORRECT_ANSWER(23) = 1;
QUESTIONS{24} = 'What did Arthur tell Lee in the second phone call?';
POSSIBLE_ANSWERS{24,1} = 'That Joanie just came home';
POSSIBLE_ANSWERS{24,2} = 'That he wanted to come over';
CORRECT_ANSWER(24) = 1;
QUESTIONS{25} = 'How did Lee react?';
POSSIBLE_ANSWERS{25,1} = 'He seemed troubled';
POSSIBLE_ANSWERS{25,2} = 'He was relieved';
CORRECT_ANSWER(25) = 1;
QUESTIONS{26} = 'What did Arthur say - Why did Joanie come back home so late?';
POSSIBLE_ANSWERS{26,1} = 'She went to drink and help her friends';
POSSIBLE_ANSWERS{26,2} = 'She was with another man';
CORRECT_ANSWER(26) = 1;
QUESTIONS{27} = 'What did you think - Why did Joanie come back home so late?';
POSSIBLE_ANSWERS{27,1} = 'She was with another man';
POSSIBLE_ANSWERS{27,2} = 'She went to drink and help her friends';
CONTEXT_ANSWERS{27,1} = 1;
CONTEXT_ANSWERS{27,2} = 2;
QUESTIONS{28} = 'Why do you think Lee reacted that way?';
POSSIBLE_ANSWERS{28,1} = 'He thought Arthur was lying';
POSSIBLE_ANSWERS{28,2} = 'He realized Arthur was having one of his paranoid episodes again';
CONTEXT_ANSWERS{28,1}  = 1;
CONTEXT_ANSWERS{28,2} = 2;
QUESTIONS{29} = 'Did you believe Arthur? Did you think Joanie really came back home?';
POSSIBLE_ANSWERS{29,1} = 'No';
POSSIBLE_ANSWERS{29,2} = 'Yes';
CONTEXT_ANSWERS{29,1} = 1;
CONTEXT_ANSWERS{29,2} = 2;
QUESTIONS{30} = 'If you didn''t believe Arthur, why do you think he lied about Joanie coming back home?';
POSSIBLE_ANSWERS{30,1} = 'He wanted to test Lee''s reaction';
POSSIBLE_ANSWERS{30,2} = 'He wanted to protect his image';
CONTEXT_ANSWERS{30,1} = 1;
CONTEXT_ANSWERS{30,2} = 2;
QUESTIONS{31} = 'Arthur suggested they will leave New York, and move to where?';
POSSIBLE_ANSWERS{31,1} = 'Connecticut';
POSSIBLE_ANSWERS{31,2} = 'Philadelphia';
CORRECT_ANSWER(31) = 1;
QUESTIONS{32} = 'How did Lee end the conversation?';
POSSIBLE_ANSWERS{32,1} = 'He said he had a headache';
POSSIBLE_ANSWERS{32,2} = 'He said he was tired';
CORRECT_ANSWER(32) = 1; % CHECK
QUESTIONS{33} = 'What was the atmosphere between Lee and the girl when the conversation ends?';
POSSIBLE_ANSWERS{33,1} = 'Tense';
POSSIBLE_ANSWERS{33,2} = 'Calm';
CORRECT_ANSWER(33) = 1;
QUESTIONS{34} = 'When you heard the phone conversation, did you think Arthur suspected Joanie was with Lee?';
POSSIBLE_ANSWERS{34,1} = 'Yes';
POSSIBLE_ANSWERS{34,2} = 'No';
CONTEXT_ANSWERS{34,1} = 1;
CONTEXT_ANSWERS{34,2} = 2;
QUESTIONS{35} = 'Did you think Joanie was cheating on Arthur?';
POSSIBLE_ANSWERS{35,1} = 'Yes';
POSSIBLE_ANSWERS{35,2} = 'No';
CONTEXT_ANSWERS{35,1} = 1;
CONTEXT_ANSWERS{35,2} = 2;
QUESTIONS{36} = 'If you did think she was cheating on him, with whom?';
POSSIBLE_ANSWERS{36,1} = 'Lee';
POSSIBLE_ANSWERS{36,2} = 'Another man';
CONTEXT_ANSWERS{36,1} = 1;
CONTEXT_ANSWERS{36,2} = 2;
QUESTIONS{37} = 'When the phone rang at the first time, why did you think the gray-haired man asked the girl if she would rather he didn''t answer it?';
POSSIBLE_ANSWERS{37,1} = 'Because they were afraid it was her husband';
POSSIBLE_ANSWERS{37,2} = 'Because they were desperate to go to sleep';
CONTEXT_ANSWERS{37,1} = 1;
CONTEXT_ANSWERS{37,2} = 2;
QUESTIONS{38} = 'Why do you think Lee didn''t tell Arthur that there was a girl at his place?';
POSSIBLE_ANSWERS{38,1} = 'He didn''t want Arthur to suspect anything';
POSSIBLE_ANSWERS{38,2} = 'He didn''t want Arthur to feel that he is interrupting';
CONTEXT_ANSWERS{38,1} = 1;
CONTEXT_ANSWERS{38,2} = 2;
QUESTIONS{39} = 'Why do you think Lee didn''t want Arthur to come over?';
POSSIBLE_ANSWERS{39,1} = 'Because Joanie was there';
POSSIBLE_ANSWERS{39,2} = 'Because he was with his girlfriend, and he didn''t want to be interrupted';
CONTEXT_ANSWERS{39,1} = 1;
CONTEXT_ANSWERS{39,2} = 2;
RATINGS{1} = 'Empathized with Arthur';
RATINGS{2} = 'Emphathized with Lee';
RATINGS{3} = 'Empathized with Joanie';
RATINGS{4} = 'Empathized with the girl';
RATINGS{5} = 'Enjoyed the story';
RATINGS{6} = 'Felt that you were engaged with the story';
RATINGS{7} = 'Felt that the neurofeedback helped you';
RATINGS{8} = 'Are certain that your interpretation is correct';
RATINGS{9} = 'Were sleepy in the scanner';

%%
% find newest file
subject = 46;
bids_id = sprintf('%03d', subject);
file_dir = ['data/sub-' bids_id '/'];
fn = findNewestFile(file_dir, [file_dir '/responses_20*']);
z = load(fn);
ratings = z.stim.rating_resps;
ntrials = length(ratings);
for t = 1:ntrials
    key_rating(t) = find(ratings(t) == all_responses);
    RATINGS{t}
    key_rating(t)
end


actual_resp = z.stim.actual_resp;
ntrials = length(actual_resp);
for t = 1:ntrials
   t;
    QUESTIONS{t};
   POSSIBLE_ANSWERS{t,actual_resp(t)}; 
end
%% analyze rating scores 
% calculate comprehension score
nQ = 39;
all_questions = 1:nQ;
comprehension_questions = 1:nQ;
comprehension_questions(CONTEXT_QUESTIONS) = [];
n_comprehension = length(comprehension_questions);
for q = 1:n_comprehension
   question = comprehension_questions(q); 
   response = z.stim.actual_resp(question);
   if response == CORRECT_ANSWER(question)
       score(q) = 1;
   else
       score(q) = 0;
   end
end
story_score = mean(score);
fprintf('mean comphrehension is %2.2f\n',  story_score)
%% now analyze cheating vs. paranoid responses
n_context = length(CONTEXT_QUESTIONS);
context_score = 0;

for q = 1:n_context
   question = CONTEXT_QUESTIONS(q);
   QUESTIONS{CONTEXT_QUESTIONS(q)};
   response = z.stim.actual_resp(question);
   POSSIBLE_ANSWERS{CONTEXT_QUESTIONS(q),response};
   if response == 1 % then cheating response
       context_score = context_score + 1;
   elseif response == 2 % then paranoid response
       context_score = context_score - 1;
   end
end
mean_context_score = context_score/n_context;
fprintf('context score for -1 = paranoid and + 1 = cheating is %2.2f\n',  mean_context_score)
%% save responses
filename=['data/sub-' bids_id '/responses_scored.mat'];
save(filename, 'story_score', 'mean_context_score', 'key_rating')
