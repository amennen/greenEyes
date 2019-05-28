KbName('UnifyKeyNames');

ONE = KbName('1!');
TWO = KbName('2@');
THREE = KbName('3#');
FOUR = KbName('4$');
FIVE = KbName('5%');
all_responses = [ONE TWO THREE FOUR FIVE];

UP_key = KbName('UpArrow');
DOWN_key = KbName('DownArrow');
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
QUESTIONS{6} = 'Why did Arthur call Lee?';
POSSIBLE_ANSWERS{6,1} = 'Because Joanie didn''t come home, and he wanted to know if Lee noticed when she left';
POSSIBLE_ANSWERS{6,2} = 'Because Joanie didn''t come home, and he wanted to know if she was at Lee''s house';
CORRECT_ANSWER(6) = 0;
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
QUESTIONS{20} = 'What did Lee answer?';
POSSIBLE_ANSWERS{20,1} = 'That it was better for Arthur to be there, when Joanie comes back home';
POSSIBLE_ANSWERS{20,2} = 'That he was tired, and rather go to sleep';
QUESTIONS{21} = 'Did Lee want Arthur to come over?';
POSSIBLE_ANSWERS{21,1} = 'No';
POSSIBLE_ANSWERS{21,2} = 'Yes';
QUESTIONS{22} = 'What did the girl ask after Lee hung up?';
POSSIBLE_ANSWERS{22,1} = '"What did he say?"';
POSSIBLE_ANSWERS{22,2} = '"What''s wrong with him?"';
QUESTIONS{23} = 'What did the girl think about how Lee handled the situation?';
POSSIBLE_ANSWERS{23,1} = 'That he handled it very well';
POSSIBLE_ANSWERS{23,2} = 'That he should have been more assertive';
QUESTIONS{24} = 'What did Arthur tell Lee in the second phone call?';
POSSIBLE_ANSWERS{24,1} = 'That Joanie just came home';
POSSIBLE_ANSWERS{24,2} = 'That he wanted to come over';
QUESTIONS{25} = 'How did Lee react?';
POSSIBLE_ANSWERS{25,1} = 'He seemed troubled';
POSSIBLE_ANSWERS{25,2} = 'He was relieved';
QUESTIONS{26} = 'What did Arthur say - Why did Joanie come back home so late?';
POSSIBLE_ANSWERS{26,1} = 'She went to drink and help her friends';
POSSIBLE_ANSWERS{26,2} = 'She was with another man';
QUESTIONS{27} = 'What did you think - Why did Joanie come back home so late?';
POSSIBLE_ANSWERS{27,1} = 'She was with another man';
POSSIBLE_ANSWERS{27,2} = 'She went to drink and help her friends';
QUESTIONS{28} = 'Why do you think Lee reacted that way?';
POSSIBLE_ANSWERS{28,1} = 'He thought Arthur was lying';
POSSIBLE_ANSWERS{28,2} = 'He realized Arthur was having one of his paranoid episodes again';
QUESTIONS{29} = 'Did you believe Arthur? Did you think Joanie really came back home?';
POSSIBLE_ANSWERS{29,1} = 'No';
POSSIBLE_ANSWERS{29,2} = 'Yes';
QUESTIONS{30} = 'If you didn''t believe Arthur, why do you think he lied about Joanie coming back home?';
POSSIBLE_ANSWERS{30,1} = 'He wanted to test Lee''s reaction';
POSSIBLE_ANSWERS{30,2} = 'He wanted to protect his image';
QUESTIONS{31} = 'Arthur suggested they will leave New York, and move to where?';
POSSIBLE_ANSWERS{31,1} = 'Connecticut';
POSSIBLE_ANSWERS{31,2} = 'Philadelphia';
QUESTIONS{32} = 'How did Lee end the conversation?';
POSSIBLE_ANSWERS{32,1} = 'He said he had a headache';
POSSIBLE_ANSWERS{32,2} = 'He said he was tired';
QUESTIONS{33} = 'What was the atmosphere between Lee and the girl when the conversation ends?';
POSSIBLE_ANSWERS{33,1} = 'Tense';
POSSIBLE_ANSWERS{33,2} = 'Calm';
QUESTIONS{34} = 'When you heard the phone conversation, did you think Arthur suspected Joanie was with Lee?';
POSSIBLE_ANSWERS{34,1} = 'Yes';
POSSIBLE_ANSWERS{34,2} = 'No';
QUESTIONS{35} = 'Did you think Joanie was cheating on Arthur?';
POSSIBLE_ANSWERS{35,1} = 'Yes';
POSSIBLE_ANSWERS{35,2} = 'No';
QUESTIONS{36} = 'If you did think she was cheating on him, with whom?';
POSSIBLE_ANSWERS{36,1} = 'Lee';
POSSIBLE_ANSWERS{36,2} = 'Another man';
QUESTIONS{37} = 'When the phone rang at the first time, why did you think the gray-haired man asked the girl if she would rather he didn''t answer it?';
POSSIBLE_ANSWERS{37,1} = 'Because they were afraid it was her husband';
POSSIBLE_ANSWERS{37,2} = 'Because they were desperate to go to sleep';
QUESTIONS{38} = 'Why do you think Lee didn''t tell Arthur that there was a girl at his place?';
POSSIBLE_ANSWERS{38,1} = 'He didn''t want Arthur to suspect anything';
POSSIBLE_ANSWERS{38,2} = 'He didn''t want Arthur to feel that he is interrupting';
QUESTIONS{39} = 'Why do you think Lee didn''t want Arthur to come over?';
POSSIBLE_ANSWERS{39,1} = 'Because Joanie was there';
POSSIBLE_ANSWERS{39,2} = 'Because he was with his girlfriend, and he didn''t want to be interrupted';
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

z = load('data/sub-102/responses_20190523T104457.mat')

ratings = z.stim.rating_resps;
ntrials = length(ratings);
for t = 1:ntrials
    key_rating(t) = find(ratings(t) == all_responses);
    t
    RATINGS{t}
    key_rating(t)
end


%%
actual_resp = z.stim.actual_resp;
ntrials = length(actual_resp);
for t = 1:ntrials
   t
    QUESTIONS{t}
   POSSIBLE_ANSWERS{t,actual_resp(t)} 
end