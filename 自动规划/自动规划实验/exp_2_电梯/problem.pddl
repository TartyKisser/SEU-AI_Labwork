(define (problem transport-scenario) (:domain lift-transport)
    (:objects
        lift_a lift_b - lift
        alex beth charlie - person
    )
    (:init
        (= (lift_position lift_a) 3)
        (= (target_floor lift_a) 3)
        (= (lift_position lift_b) 2)
        (= (target_floor lift_b) 2)
        (= (person_location alex) 1)
        (= (destination alex) 4)
        (= (person_location beth) 3)
        (= (destination beth) 2)
        (= (person_location charlie) 5)
        (= (destination charlie) 2)
        (= (top_floor) 5)
        (= (total_cost) 0)
    )
    (:goal (and
            (goal_reached alex)
            (goal_reached beth)
            (goal_reached charlie)
        )
    )
    (:metric minimize (total_cost))
)