(define (domain vehicle_parking)
    (:requirements :typing :negative-preconditions :disjunctive-preconditions)
    (:types vehicle garage slot - object)
    (:predicates
        (located ?vehicle - vehicle ?garage - garage ?slot - slot)
        (occupied ?garage - garage ?slot - slot)
        (adjacent_slot ?lower ?upper - slot)
        (top_slot ?slot - slot)
    )
    (:action transfer_between_garages
        :parameters (
            ?vehicle - vehicle
            ?source_garage ?target_garage - garage
            ?slot - slot
        )
        :precondition (and
            (top_slot ?slot)
            (located ?vehicle ?source_garage ?slot)
            (occupied ?source_garage ?slot)
            (not (occupied ?target_garage ?slot))
        )
        :effect (and 
            (not (located ?vehicle ?source_garage ?slot))
            (not (occupied ?source_garage ?slot))
            (located ?vehicle ?target_garage ?slot)
            (occupied ?target_garage ?slot)
        )
    )
    (:action shift_within_garage
        :parameters (
            ?vehicle - vehicle
            ?garage - garage
            ?current_slot ?new_slot - slot
        )
        :precondition (and
            (or
                (adjacent_slot ?current_slot ?new_slot)
                (adjacent_slot ?new_slot ?current_slot)
            )
            (located ?vehicle ?garage ?current_slot)
            (occupied ?garage ?current_slot)
            (not (occupied ?garage ?new_slot))
        )
        :effect (and 
            (not (located ?vehicle ?garage ?current_slot))
            (not (occupied ?garage ?current_slot))
            (located ?vehicle ?garage ?new_slot)
            (occupied ?garage ?new_slot)
        )
    )
)