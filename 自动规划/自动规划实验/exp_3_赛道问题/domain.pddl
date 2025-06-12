(define (domain racetrack-advanced)
  (:requirements :strips)
  
  (:predicates
    (car-at ?pos)
    (car-vel ?vx ?vy)
    (can-move ?from ?to ?vx ?vy)
    (can-adjust-vel ?old-vx ?new-vx ?old-vy ?new-vy)
    (wall ?pos)
    (goal ?pos)
    (zero-vel ?v)
  )
  
  (:action adjust-velocity
    :parameters (?pos ?old-vx ?new-vx ?old-vy ?new-vy)
    :precondition (and
      (car-at ?pos)
      (car-vel ?old-vx ?old-vy)
      (can-adjust-vel ?old-vx ?new-vx ?old-vy ?new-vy)
    )
    :effect (and
      (not (car-vel ?old-vx ?old-vy))
      (car-vel ?new-vx ?new-vy)
    )
  )
  
  (:action move-car
    :parameters (?from ?to ?vx ?vy)
    :precondition (and
      (car-at ?from)
      (car-vel ?vx ?vy)
      (can-move ?from ?to ?vx ?vy)
      (not (wall ?to))
    )
    :effect (and
      (not (car-at ?from))
      (car-at ?to)
    )
  )
)