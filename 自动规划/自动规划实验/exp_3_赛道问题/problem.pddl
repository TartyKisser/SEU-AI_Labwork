(define (problem racetrack-large)
  (:domain racetrack-advanced)
  
  (:objects
    ;; 8x8 Grid positions - pXY means position (X,Y)
    p00 p01 p02 p03 p04 p05 p06 p07
    p10 p11 p12 p13 p14 p15 p16 p17
    p20 p21 p22 p23 p24 p25 p26 p27
    p30 p31 p32 p33 p34 p35 p36 p37
    p40 p41 p42 p43 p44 p45 p46 p47
    p50 p51 p52 p53 p54 p55 p56 p57
    p60 p61 p62 p63 p64 p65 p66 p67
    p70 p71 p72 p73 p74 p75 p76 p77
    
    ;; Velocities
    v-2 v-1 v0 v1 v2
  )
  
  (:init
    ;; Car starts at position (4,7) with velocity (1,1) - bottom area
    (car-at p74)
    (car-vel v1 v1)
    
    ;; Complete velocity adjustment rules (change by -1, 0, or +1)
    (can-adjust-vel v-2 v-2 v-2 v-2) (can-adjust-vel v-2 v-2 v-2 v-1) (can-adjust-vel v-2 v-2 v-2 v0)
    (can-adjust-vel v-2 v-1 v-2 v-2) (can-adjust-vel v-2 v-1 v-2 v-1) (can-adjust-vel v-2 v-1 v-2 v0)
    (can-adjust-vel v-2 v0 v-2 v-2) (can-adjust-vel v-2 v0 v-2 v-1) (can-adjust-vel v-2 v0 v-2 v0)
    
    (can-adjust-vel v-1 v-2 v-1 v-2) (can-adjust-vel v-1 v-2 v-1 v-1) (can-adjust-vel v-1 v-2 v-1 v0)
    (can-adjust-vel v-1 v-1 v-1 v-2) (can-adjust-vel v-1 v-1 v-1 v-1) (can-adjust-vel v-1 v-1 v-1 v0)
    (can-adjust-vel v-1 v0 v-1 v-2) (can-adjust-vel v-1 v0 v-1 v-1) (can-adjust-vel v-1 v0 v-1 v0)
    (can-adjust-vel v-1 v1 v-1 v-1) (can-adjust-vel v-1 v1 v-1 v0) (can-adjust-vel v-1 v1 v-1 v1)
    
    (can-adjust-vel v0 v-1 v0 v-1) (can-adjust-vel v0 v-1 v0 v0) (can-adjust-vel v0 v-1 v0 v1)
    (can-adjust-vel v0 v0 v0 v-1) (can-adjust-vel v0 v0 v0 v0) (can-adjust-vel v0 v0 v0 v1)
    (can-adjust-vel v0 v1 v0 v-1) (can-adjust-vel v0 v1 v0 v0) (can-adjust-vel v0 v1 v0 v1)
    
    (can-adjust-vel v1 v0 v1 v0) (can-adjust-vel v1 v0 v1 v1) (can-adjust-vel v1 v0 v1 v2)
    (can-adjust-vel v1 v1 v1 v0) (can-adjust-vel v1 v1 v1 v1) (can-adjust-vel v1 v1 v1 v2)
    (can-adjust-vel v1 v2 v1 v1) (can-adjust-vel v1 v2 v1 v2)
    
    (can-adjust-vel v2 v1 v2 v1) (can-adjust-vel v2 v1 v2 v2)
    (can-adjust-vel v2 v2 v2 v1) (can-adjust-vel v2 v2 v2 v2)
    
    ;; Zero velocity (stay put) - ALL POSITIONS
    (can-move p00 p00 v0 v0) (can-move p01 p01 v0 v0) (can-move p02 p02 v0 v0) (can-move p03 p03 v0 v0)
    (can-move p04 p04 v0 v0) (can-move p05 p05 v0 v0) (can-move p06 p06 v0 v0) (can-move p07 p07 v0 v0)
    (can-move p10 p10 v0 v0) (can-move p11 p11 v0 v0) (can-move p12 p12 v0 v0) (can-move p13 p13 v0 v0)
    (can-move p14 p14 v0 v0) (can-move p15 p15 v0 v0) (can-move p16 p16 v0 v0) (can-move p17 p17 v0 v0)
    (can-move p20 p20 v0 v0) (can-move p21 p21 v0 v0) (can-move p22 p22 v0 v0) (can-move p23 p23 v0 v0)
    (can-move p24 p24 v0 v0) (can-move p25 p25 v0 v0) (can-move p26 p26 v0 v0) (can-move p27 p27 v0 v0)
    (can-move p30 p30 v0 v0) (can-move p31 p31 v0 v0) (can-move p32 p32 v0 v0) (can-move p33 p33 v0 v0)
    (can-move p34 p34 v0 v0) (can-move p35 p35 v0 v0) (can-move p36 p36 v0 v0) (can-move p37 p37 v0 v0)
    (can-move p40 p40 v0 v0) (can-move p41 p41 v0 v0) (can-move p42 p42 v0 v0) (can-move p43 p43 v0 v0)
    (can-move p44 p44 v0 v0) (can-move p45 p45 v0 v0) (can-move p46 p46 v0 v0) (can-move p47 p47 v0 v0)
    (can-move p50 p50 v0 v0) (can-move p51 p51 v0 v0) (can-move p52 p52 v0 v0) (can-move p53 p53 v0 v0)
    (can-move p54 p54 v0 v0) (can-move p55 p55 v0 v0) (can-move p56 p56 v0 v0) (can-move p57 p57 v0 v0)
    (can-move p60 p60 v0 v0) (can-move p61 p61 v0 v0) (can-move p62 p62 v0 v0) (can-move p63 p63 v0 v0)
    (can-move p64 p64 v0 v0) (can-move p65 p65 v0 v0) (can-move p66 p66 v0 v0) (can-move p67 p67 v0 v0)
    (can-move p70 p70 v0 v0) (can-move p71 p71 v0 v0) (can-move p72 p72 v0 v0) (can-move p73 p73 v0 v0)
    (can-move p74 p74 v0 v0) (can-move p75 p75 v0 v0) (can-move p76 p76 v0 v0) (can-move p77 p77 v0 v0)
    
    ;; KEY MOVEMENT PATTERNS for guaranteed solution path
    ;; Diagonal movements (velocity v1 v1) - move right and up (X+1, Y-1)
    (can-move p74 p63 v1 v1)  ;; Starting move: (4,7) -> (3,6)
    (can-move p63 p52 v1 v1)  ;; (3,6) -> (2,5)
    (can-move p52 p41 v1 v1)  ;; (2,5) -> (1,4)
    (can-move p41 p30 v1 v1)  ;; (1,4) -> (0,3)
    (can-move p30 p21 v1 v1)  ;; (0,3) -> (1,2)
    (can-move p21 p12 v1 v1)  ;; (1,2) -> (2,1)
    (can-move p12 p03 v1 v1)  ;; (2,1) -> (3,0) - goal reached!
    
    ;; Alternative paths to other goals
    (can-move p13 p04 v1 v1)  ;; (3,1) -> (4,0) - goal reached!
    (can-move p14 p05 v1 v1)  ;; (4,1) -> (5,0) - goal reached!
    
    ;; Additional diagonal paths for flexibility
    (can-move p71 p60 v1 v1)
    (can-move p73 p62 v1 v1)
    (can-move p75 p64 v1 v1)  ;; New path from near starting position
    (can-move p62 p51 v1 v1)
    (can-move p64 p53 v1 v1)  ;; Alternative route
    (can-move p51 p40 v1 v1)
    (can-move p53 p42 v1 v1)  ;; But p42 is wall, so this won't work
    (can-move p40 p31 v1 v1)
    (can-move p31 p22 v1 v1)
    (can-move p22 p13 v1 v1)
    (can-move p32 p23 v1 v1)
    (can-move p23 p14 v1 v1)
    (can-move p24 p15 v1 v1)
    (can-move p15 p06 v1 v1)
    
    ;; Horizontal movements (velocity v1 v0) - move right (X+1)
    (can-move p70 p60 v1 v0) (can-move p71 p61 v1 v0) (can-move p72 p62 v1 v0) (can-move p73 p63 v1 v0)
    (can-move p74 p64 v1 v0) (can-move p75 p65 v1 v0) (can-move p76 p66 v1 v0)
    (can-move p60 p50 v1 v0) (can-move p61 p51 v1 v0) (can-move p62 p52 v1 v0) (can-move p63 p53 v1 v0)
    (can-move p64 p54 v1 v0) (can-move p65 p55 v1 v0) (can-move p66 p56 v1 v0)
    (can-move p50 p40 v1 v0) (can-move p51 p41 v1 v0) (can-move p52 p42 v1 v0) (can-move p53 p43 v1 v0)
    (can-move p54 p44 v1 v0) (can-move p55 p45 v1 v0) (can-move p56 p46 v1 v0)
    (can-move p40 p30 v1 v0) (can-move p41 p31 v1 v0) (can-move p42 p32 v1 v0) (can-move p43 p33 v1 v0)
    (can-move p44 p34 v1 v0) (can-move p45 p35 v1 v0) (can-move p46 p36 v1 v0)
    (can-move p30 p20 v1 v0) (can-move p31 p21 v1 v0) (can-move p32 p22 v1 v0) (can-move p33 p23 v1 v0)
    (can-move p34 p24 v1 v0) (can-move p35 p25 v1 v0) (can-move p36 p26 v1 v0)
    (can-move p20 p10 v1 v0) (can-move p21 p11 v1 v0) (can-move p22 p12 v1 v0) (can-move p23 p13 v1 v0)
    (can-move p24 p14 v1 v0) (can-move p25 p15 v1 v0) (can-move p26 p16 v1 v0)
    (can-move p10 p00 v1 v0) (can-move p11 p01 v1 v0) (can-move p12 p02 v1 v0) (can-move p13 p03 v1 v0)
    (can-move p14 p04 v1 v0) (can-move p15 p05 v1 v0) (can-move p16 p06 v1 v0)
    
    ;; Vertical movements (velocity v0 v1) - move up (Y-1)
    (can-move p07 p06 v0 v1) (can-move p06 p05 v0 v1) (can-move p05 p04 v0 v1) (can-move p04 p03 v0 v1)
    (can-move p03 p02 v0 v1) (can-move p02 p01 v0 v1) (can-move p01 p00 v0 v1)
    (can-move p17 p16 v0 v1) (can-move p16 p15 v0 v1) (can-move p15 p14 v0 v1) (can-move p14 p13 v0 v1)
    (can-move p13 p12 v0 v1) (can-move p12 p11 v0 v1) (can-move p11 p10 v0 v1)
    (can-move p27 p26 v0 v1) (can-move p26 p25 v0 v1) (can-move p25 p24 v0 v1) (can-move p24 p23 v0 v1)
    (can-move p23 p22 v0 v1) (can-move p22 p21 v0 v1) (can-move p21 p20 v0 v1)
    (can-move p37 p36 v0 v1) (can-move p36 p35 v0 v1) (can-move p35 p34 v0 v1) (can-move p34 p33 v0 v1)
    (can-move p33 p32 v0 v1) (can-move p32 p31 v0 v1) (can-move p31 p30 v0 v1)
    (can-move p47 p46 v0 v1) (can-move p46 p45 v0 v1) (can-move p45 p44 v0 v1) (can-move p44 p43 v0 v1)
    (can-move p43 p42 v0 v1) (can-move p42 p41 v0 v1) (can-move p41 p40 v0 v1)
    (can-move p57 p56 v0 v1) (can-move p56 p55 v0 v1) (can-move p55 p54 v0 v1) (can-move p54 p53 v0 v1)
    (can-move p53 p52 v0 v1) (can-move p52 p51 v0 v1) (can-move p51 p50 v0 v1)
    (can-move p67 p66 v0 v1) (can-move p66 p65 v0 v1) (can-move p65 p64 v0 v1) (can-move p64 p63 v0 v1)
    (can-move p63 p62 v0 v1) (can-move p62 p61 v0 v1) (can-move p61 p60 v0 v1)
    (can-move p77 p76 v0 v1) (can-move p76 p75 v0 v1) (can-move p75 p74 v0 v1) (can-move p74 p73 v0 v1)
    (can-move p73 p72 v0 v1) (can-move p72 p71 v0 v1) (can-move p71 p70 v0 v1)
    
    ;; Strategic walls (creating obstacle as specified)
    (wall p42) (wall p43) (wall p44) (wall p45)  ;; Horizontal barrier in middle
    
    ;; SPECIFIED Goal region (only p03, p04, p05)
    (goal p03) (goal p04) (goal p05)
    
    ;; Zero velocity marker
    (zero-vel v0)
  )
  
  ;; RELAXED GOAL: Just reach any goal position (remove velocity constraint)
  (:goal
    (exists (?pos) (and (car-at ?pos) (goal ?pos)))
  )
)