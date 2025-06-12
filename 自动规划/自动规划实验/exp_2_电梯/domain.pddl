(define (domain lift-transport)
    (:requirements :typing :conditional-effects :disjunctive-preconditions :fluents :negative-preconditions)
    (:types lift person - object)
    (:predicates
        (inside ?person - person ?lift - lift)
        (goal_reached ?person - person)
    )
    (:functions
        (person_location ?person - person) - number
        (lift_position ?lift - lift) - number
        (destination ?person - person) - number
        (target_floor ?lift - lift) - number
        (top_floor) - number
        (total_cost) - number
    )
    (:action board
        :parameters (?person - person ?lift - lift)
        :precondition (and
            (= (lift_position ?lift) (person_location ?person))
            (forall (?l - lift) (not (inside ?person ?l)))
            (not (goal_reached ?person))
        )
        :effect (inside ?person ?lift)
    )
    (:action exit
        :parameters (?person - person ?lift - lift)
        :precondition (and
            (inside ?person ?lift)
            (= (person_location ?person) (destination ?person))
        )
        :effect (and
            (not (inside ?person ?lift))
            (goal_reached ?person)
        )
    )
    (:action ascend
        :parameters (?lift - lift)
        :precondition (and
            (or
                (exists (?person - person) (and
                    (not (inside ?person ?lift))
                    (< (lift_position ?lift) (person_location ?person))
                ))
                (exists (?person - person) (and
                    (inside ?person ?lift)
                    (< (lift_position ?lift) (destination ?person))
                ))
                (< (lift_position ?lift) (target_floor ?lift))
            )
            (< (lift_position ?lift) (top_floor))
        )
        :effect (and
            (increase (lift_position ?lift) 1)
            (forall (?person - person)
                (when (inside ?person ?lift) 
                    (increase (person_location ?person) 1)
                )
            )
            (forall (?person - person)
                (when (inside ?person ?lift) 
                    (increase (total_cost) 20)
                )
            )
            (increase (total_cost) 1)
        )
    )
    (:action descend
        :parameters (?lift - lift)
        :precondition (and
            (or
                (exists (?person - person) (and
                    (not (inside ?person ?lift))
                    (> (lift_position ?lift) (person_location ?person))
                ))
                (exists (?person - person) (and
                    (inside ?person ?lift)
                    (> (lift_position ?lift) (destination ?person))
                ))
                (> (lift_position ?lift) (target_floor ?lift))
            )
            (>= (lift_position ?lift) 1)
        )
        :effect (and
            (decrease (lift_position ?lift) 1)
            (forall (?person - person)
                (when (inside ?person ?lift) 
                    (decrease (person_location ?person) 1)
                )
            )
            (forall (?person - person)
                (when (inside ?person ?lift) 
                    (increase (total_cost) 20)
                )
            )
            (increase (total_cost) 1)
        )
    )
)