(defun SetWidthFactor (width / ss ent)
  (setq ss (ssget "_X" '((0 . "TEXT")))) ; Select all TEXT entities
  (if ss
    (foreach ent (vl-remove-if 'null (mapcar 'cadr (ssnamex ss)))
      (entmod (subst (cons 41 width) (assoc 41 (entget ent)) (entget ent)))
      (entupd ent)
    )
  )
  (princ)
)