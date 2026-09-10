require 'digest'
require 'json'
require 'open3'

root = ARGV.fetch(0)
source_path = File.join(root, 'autograd/vjp_elementwise.go')
source = File.binread(source_path)
base, error, result = Open3.capture3('git', '-C', root, 'show', '54cc29649184d4bb22241589a1519070128d3e49:autograd/vjp_elementwise.go')
abort error unless result.success?
lines = source.lines
restored = []
proofs = []
i = 0
while i < lines.length
  match = /\A(\t+)if n >= 0 && .*\{\n\z/.match(lines[i])
  unless match
    restored << lines[i]
    i += 1
    next
  end
  indent = match[1]
  start = i
  alternate = ((i + 1)...lines.length).find { |k| lines[k] == "#{indent}} else {\n" }
  abort "missing cold path at #{i + 1}" unless alternate
  finish = ((alternate + 1)...lines.length).find { |k| lines[k] == "#{indent}}\n" }
  abort "missing proof end at #{i + 1}" unless finish
  abort 'hot path does not range over destination' unless lines[start + 2] == "#{indent}\tfor i := range ds {\n"
  hot_body = lines[(start + 3)...(alternate - 1)]
  cold_body = lines[(alternate + 2)...(finish - 1)]
  abort 'hot per-element body differs from original cold body' unless hot_body == cold_body
  cold = lines[(alternate + 1)...finish].map { |line| line.sub(/\A\t/, '') }
  abort 'cold path is not original indexed loop' unless cold.first == "#{indent}for i := 0; i < n; i++ {\n"
  restored.concat(cold)
  proofs << { guard_line: start + 1, cold_else_line: alternate + 1, end_line: finish + 1, guard: lines[start].strip }
  i = finish + 1
end
abort "expected six proof paths, got #{proofs.length}" unless proofs.length == 6
abort 'bytes outside the six guarded-loop replacements changed' unless restored.join.b == base.b
oracle = File.join(root, 'autograd/vjp_bounds_internal_test.go')
abort 'oracle changed' unless Digest::SHA256.file(oracle).hexdigest == 'bd26d0195b02e23aa1406f01f7df25fff5aa65ea81fea0c3bae9f31d3359648c'
puts JSON.pretty_generate({ source_sha256: Digest::SHA256.hexdigest(source), base_sha256: Digest::SHA256.hexdigest(base), restored_source_matches_base: true, hot_bodies_match_original_cold_bodies: true, proof_paths: proofs, oracle_sha256: Digest::SHA256.file(oracle).hexdigest })
