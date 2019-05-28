%% specify any input - run from local directory
debug = 0;
subjectNum = 102;
bidsId = sprintf('sub-%03d',subjectNum);

saveDir = ['data/' bidsId];
if ~exist(saveDir)
    mkdir(saveDir)
end
ListenChar(2)
%% SPECIFY QUESTIONS
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

% SPECIFY QUESTION TYPES
nQ = 39;
NEUTRAL = 1;
CONTEXT  = 2;
questionTypes = ones(39,1);
indContext = [5,9,27,28,29,30,34,35,36,37,38,39];
indNeutral = 1:nQ;
indNeutral(indContext) = [];
nContext = length(indContext);
questionTypes(indContext) = CONTEXT;

% RATINGS
nR = 9;

% SPECIFY CORRECT ANSWERS
%% to print answers
% q=39
% QUESTIONS{q}
% POSSIBLE_ANSWERS{q,:}

%% Specify experiment parameters
questionDur = -1; % means unlimited amount of time to answer
% display parameters
textColor = 255;
textFont = 'Arial';
textSize = 35;
textSpacing = 25;
fixColor = 0;
respColor = 255;
backColor = 127;
fixationSize = 4;% pixels
progWidth = 400; % image loading progress bar
progHeight = 20;

%ScreenResX = 1280;
%ScreenResY = 720;
DEVICE = -1;
KbName('UnifyKeyNames');
ONE = KbName('1!');
TWO = KbName('2@');
THREE = KbName('3#');
FOUR = KbName('4$');
FIVE = KbName('5%');
UP_key = KbName('UpArrow');
DOWN_key = KbName('DownArrow');

basicInstruct{1} = 'Now that you have received all your information and clues, it is time to tell us what you think happened.'; 
basicInstruct{2} = 'You will now be asked to answer questions about the story you just heard.';
basicInstruct{2} = 'Please do you best to answer the questions as accurately and honestly as possible.';
basicInstruct{3} = 'Read the questions carefully. If you respond too fast, your response won''t count and you''ll need to respond again.';
basicInstruct{4} = 'Hit the UP arrow to choose the top response, and hit the DOWN arrow to choose the bottom response.';
basicInstruct{5} = '-- Press the UP arrow when you are ready to begin. --';
allBasicInstruct = [basicInstruct{1} '\n' basicInstruct{2} '\n' basicInstruct{3} '\n' basicInstruct{4} '\n\n\n' basicInstruct{5} ];
ratingsInstruct{1} = 'Next you will provide ratings. Please use the number keys 1-5 for ratings.';
ratingsInstruct{2} = 'Read the questions carefully. If you respond too fast, your response won''t count and you''ll need to respond again.';
ratingsInstruct{3} = '-- Press the UP arrow when you are ready to begin. --';
allRatingsInstruct = [ratingsInstruct{1} '\n' ratingsInstruct{2}  '\n\n\n' ratingsInstruct{3} ];

ratingsInstruct{3} = 'Please rate on a scale from 1-5 how much you:\n(1 = the least; 5 = the most)';
%% Initalize screens
screenNumbers = Screen('Screens');
Screen('Preference', 'SkipSyncTests', 2);
if debug
    screenNum = 0;
else
    screenNum = screenNumbers(end);
end
%retrieve the size of the display screen
if debug
    screenX = 800;
    screenY = 800;
else
    % first just make the screen tiny
    
    [screenX screenY] = Screen('WindowSize',screenNum);
    otherscreen = screenNumbers(1);
    if otherscreen ~= screenNum
        % open another window
        [s2x s2y] = Screen('WindowSize', otherscreen);
        otherWindow = Screen(otherscreen,'OpenWindow',backColor);
    end
    % put this back in!!!
    windowSize.degrees = [51 30];
    resolution = Screen('Resolution', screenNum);
    %resolution = Screen('Resolution', 0); % REMOVE THIS AFTERWARDS!!
    windowSize.pixels = [resolution.width resolution.height];
    screenX = windowSize.pixels(1);
    screenY = windowSize.pixels(2);
end
% open window
if debug == 1
    mainWindow = Screen(screenNum, 'OpenWindow', backColor,[0 0 screenX screenY]);
else
    mainWindow = Screen(screenNum, 'OpenWindow', backColor);
end
ifi = Screen('GetFlipInterval', mainWindow);
slack  = ifi/2;
% details of main window
centerX = screenX/2; centerY = screenY/2;
Screen(mainWindow,'TextFont',textFont);
Screen(mainWindow,'TextSize',textSize);


%% prepare stimuli
% Questions are in a specified order, but position of the answers should be
% counterbalanced
% if not context position, then correct answer is always 1**

UP = 1;
DOWN = 2;
contextPos = [1 2 1 2 1 2 1 2 1 2 1 2];
stim.this_subjects_context_pos = contextPos(randperm(length(contextPos)));
if mod(subjectNum,2) == 0
    neutralPos = [1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1];
else
    neutralPos = [1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 2];
end
stim.this_subjects_neutral_pos = neutralPos(randperm(length(neutralPos)));
% now build all positions
stim.all_q_positions = zeros(1,nQ);
neutral_counter = 1;
context_counter = 1;
for q = 1:nQ
    if questionTypes(q) == NEUTRAL
        stim.all_q_positions(q) = stim.this_subjects_neutral_pos(neutral_counter);
        neutral_counter = neutral_counter + 1;
    elseif questionTypes(q) == CONTEXT
        stim.all_q_positions(q) = stim.this_subjects_context_pos(context_counter);
        context_counter = context_counter + 1;
    end
end

%% display
%for i=1:length(basicInstruct)
    %tempBounds = Screen('TextBounds',mainWindow,basicInstruct{i});
    DrawFormattedText(mainWindow,allBasicInstruct,'center','center',textColor,70,[],[],1.2)
    %Screen('drawtext',mainWindow,basicInstruct{i},centerX-tempBounds(3)/2,screenY*(1/3)+1.5*textSpacing*(i-1),textColor);
    %clear tempBounds;
%end
Screen('Flip',mainWindow);

FlushEvents('keyDown');
waitForKeyboard(UP_key,DEVICE);


Priority(MaxPriority(screenNum));
Screen(mainWindow,'FillRect',backColor);
timing.runStart = GetSecs;
Screen('Flip',mainWindow);
Priority(0);

%% trial sequence

stim.question_resps = NaN(nQ,1);
stim.question_rts = NaN(nQ,1);
questionHeight = centerY - centerY/2; % quarter of the way up
textSpacing=50;
answer1Height = centerY-textSpacing;
answer2Height = centerY+textSpacing;
wrapat = 70;
for iTrial = 1:nQ
    thisQuestion = QUESTIONS{iTrial};
    if stim.all_q_positions(iTrial) == 1
        stim.upResponse{iTrial} = POSSIBLE_ANSWERS{iTrial,1};
        stim.downResponse{iTrial} = POSSIBLE_ANSWERS{iTrial,2};
    elseif stim.all_q_positions(iTrial) == 2
        stim.upResponse{iTrial} = POSSIBLE_ANSWERS{iTrial,2};
        stim.downResponse{iTrial} = POSSIBLE_ANSWERS{iTrial,1};
    end
    questionText = {};
    questionText{1} = thisQuestion;
    questionText{2} = stim.upResponse{iTrial};
    questionText{3} = stim.downResponse{iTrial};
    fullQuestion = [questionText{1} '\n\n\n\n\n' questionText{2} '\n\n\n' questionText{3}];

    DrawFormattedText(mainWindow,fullQuestion, 'center', 'center', textColor, wrapat)

    timing.questionOn(iTrial) = Screen('Flip',mainWindow);
    WaitSecs(1);
    FlushEvents('keyDown');
    % now wait for response and only advance when one of the correct keys
    % is pressed
    while isnan(stim.question_resps(iTrial))
        [keyIsDown, secs, keyCode] = KbCheck(DEVICE); % -1 checks all keyboards
        if keyIsDown
            if keyCode(UP_key) + keyCode(DOWN_key) > 0 % if one of them is pressed
                %fprintf('key pressed\n')
                % record response and reaction time
                stim.question_resps(iTrial) = find(keyCode,1);
                % record which choice it was to make it easier to go back
                if stim.all_q_positions(iTrial) == 1
                    if stim.question_resps(iTrial) == UP_key
                        stim.actual_resp(iTrial) = 1;
                    else
                        stim.actual_resp(iTrial) = 2;
                    end
                else
                    if stim.question_resps(iTrial) == UP_key
                        stim.actual_resp(iTrial) = 2;
                    else
                        stim.actual_resp(iTrial) = 1;
                    end
                end
                stim.question_rts(iTrial) = secs - timing.questionOn(iTrial);
            end
        end
    end
    FlushEvents('keyDown');
    
end


Screen('FillRect',mainWindow,backColor);
Screen('Flip',mainWindow);
%% now do the ratings
DrawFormattedText(mainWindow,allRatingsInstruct,'center','center',textColor,70,[],[],1.2)
Screen('Flip',mainWindow);
FlushEvents('keyDown');
waitForKeyboard(UP_key,DEVICE);

stim.rating_resps = NaN(nR,1);
stim.rating_rts = NaN(nR,1);
for iTrial = 1:nR
    thisQuestion = ratingsInstruct{3};
    fullQuestion = [ratingsInstruct{3} '\n\n\n\n\n' RATINGS{iTrial}];
    DrawFormattedText(mainWindow,fullQuestion, 'center', 'center', textColor, wrapat);

   timing.ratingOn(iTrial) = Screen('Flip',mainWindow);
    WaitSecs(1);
    FlushEvents('keyDown');
    % now wait for response and only advance when one of the correct keys
    % is pressed
    while isnan(stim.rating_resps(iTrial))
        [keyIsDown, secs, keyCode] = KbCheck(DEVICE); % -1 checks all keyboards
        if keyIsDown
            if keyCode(ONE) + keyCode(TWO) + keyCode(THREE) + keyCode(FOUR) + keyCode(FIVE) > 0 % if one of them is pressed
                fprintf('key pressed\n')
                % record response and reaction time
                stim.rating_resps(iTrial) = find(keyCode,1);
                stim.rating_rts(iTrial) = secs - timing.ratingOn(iTrial);
            end
        end
    end
    FlushEvents('keyDown');
    
end

%% save
save([saveDir '/responses' '_' datestr(now,30)],'stim', 'timing');
DrawFormattedText(mainWindow,'Thank you!', 'center', 'center', textColor, wrapat);
Screen('Flip',mainWindow);
WaitSecs(3);
% clean up and go home
sca;
ListenChar(1);
fclose('all');

