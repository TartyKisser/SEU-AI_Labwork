 (define (domain temporal_elevators)
    (:requirements :typing :fluents :durative-actions)
    (:types elevator passenger num - object)
    (:predicates
        (passenger-at ?person - passenger ?floor - num)
        (boarded ?person - passenger ?lift - elevator)
        (lift-at ?lift - elevator ?floor - num)
        (next ?n1 - num ?n2 - num)
    )
    (:functions
        (person_speed ?person - passenger)
        (elevator_speed ?lift - elevator)
        (floor_distance ?f1 ?f2 - num)
    )
    (:durative-action move-up
        :parameters (?lift - elevator ?cur ?nxt - num)
        :duration (= ?duration
            (/ (floor_distance ?cur ?nxt) (elevator_speed ?lift))
        )
        :condition (and
            (at start (lift-at ?lift ?cur))
            (over all (next ?cur ?nxt))
        )
        :effect (and
            (at start (not (lift-at ?lift ?cur)))
            (at end (lift-at ?lift ?nxt))
        )
    )
    (:durative-action move-down
        :parameters (?lift - elevator ?cur ?nxt - num)
        :duration (= ?duration
            (/ (floor_distance ?cur ?nxt) (elevator_speed ?lift))
        )
        :condition (and
            (at start (lift-at ?lift ?cur))
            (over all (next ?nxt ?cur))
        )
        :effect (and
            (at start (not (lift-at ?lift ?cur)))
            (at end (lift-at ?lift ?nxt))
        )
    )
    (:durative-action board
        :parameters (?per - passenger ?flr - num ?lift - elevator)
        :duration (= ?duration (person_speed ?per))
        :condition (and
            (over all (lift-at ?lift ?flr))
            (at start (passenger-at ?per ?flr))
        )
        :effect (and
            (at start (not (passenger-at ?per ?flr)))
            (at end (boarded ?per ?lift))
        )
    )
    (:durative-action leave
        :parameters (?per - passenger ?flr - num ?lift - elevator)
        :duration (= ?duration (person_speed ?per))
        :condition (and
            (over all (lift-at ?lift ?flr))
            (at start (boarded ?per ?lift))
        )
        :effect (and
            (at end (passenger-at ?per ?flr))
            (at start (not (boarded ?per ?lift)))
        )
    )
)