# Frozen-protocol validation and exact rank-test summary; Ruby standard library.
# Usage: ruby analyze.rb paired.txt [controls.txt] > analysis.csv
require 'csv'

def median(values)
  sorted = values.sort
  sorted[sorted.length / 2]
end

# Two-sided exact permutation Mann-Whitney test, including average tied ranks.
# Seven samples per arm give C(14, 7) = 3432 equally likely assignments.
def rank_p(old, candidate)
  values = old + candidate
  return 1.0 if values.uniq.length == 1
  ranks = values.map do |value|
    lower = values.count { |v| v < value }
    equal = values.count(value)
    lower + (equal + 1) / 2.0
  end
  center = 7 * 15 / 2.0
  distance = (ranks.first(7).sum - center).abs
  extreme = (0...14).to_a.combination(7).count do |indexes|
    (indexes.sum { |i| ranks[i] } - center).abs >= distance - 1e-12
  end
  extreme / 3432.0
end

def expected_names(controls)
  return %w[BenchmarkSiLUBackwardF64_256K_cpu BenchmarkSigmoidF64_64K_cpu BenchmarkSoftplusF64_256K_cpu] if controls
  [2048, 262144].flat_map do |n|
    %w[active mixed].flat_map do |distribution|
      %w[forward backward].flat_map do |direction|
        %w[leaf execute].map do |boundary|
          "BenchmarkVGELUF64NeonBoundary/#{boundary}/#{direction}/#{distribution}/n#{n}"
        end
      end
    end
  end
end

def parse(path, controls)
  batches, metadata = [], {}
  hashes = {}
  File.foreach(path) do |line|
    if line =~ /\A(old|new)-sha256: ([0-9a-f]{64})\s*\z/
      hashes[Regexp.last_match(1)] = Regexp.last_match(2)
    elsif line =~ /\A(campaign|pair|procs): (\d+)\s*\z/
      metadata[Regexp.last_match(1)] = Regexp.last_match(2).to_i
    elsif line =~ /\Aarm: (old|new)\s*\z/
      batches << metadata.merge('arm' => Regexp.last_match(1), 'rows' => [], 'pass' => false)
    elsif line.start_with?('Benchmark')
      match = line.match(/\A(\S+)\s+(\d+)\s+(\S+)\s+ns\/op\s+(\d+)\s+B\/op\s+(\d+)\s+allocs\/op\s*\z/)
      raise "Malformed benchmark: #{line}" unless match && !batches.empty?
      name, iterations, ns, bytes, allocs = match.captures
      suffix = name[/-(\d+)\z/, 1]
      raise 'GOMAXPROCS suffix mismatch' unless (suffix || '1').to_i == batches.last['procs']
      raise 'Invalid timing' unless Float(ns).finite? && Float(ns) > 0 && iterations.to_i > 0
      batches.last['rows'] << [name.sub(/-\d+\z/, ''), Float(ns), bytes.to_i, allocs.to_i]
    elsif line.strip == 'PASS'
      raise 'Unexpected PASS' if batches.empty? || batches.last['pass']
      batches.last['pass'] = true
    elsif line.include?('FAIL')
      raise "Failed invocation: #{line}"
    end
  end
  raise 'Frozen binary SHA mismatch' unless hashes == {
    'old' => 'c95691bc1a7289094bc523250fde7a9cae614756635b78701e9c306962b246cb',
    'new' => 'b768ca10228a7487aad1cbef3a658c99cb21078b5059b815fa690b0d9bcc7614'
  }
  expected = (1..3).flat_map do |campaign|
    (1..7).flat_map do |pair|
      (campaign.odd? ? [1, 12] : [12, 1]).flat_map do |procs|
        ((pair + campaign).odd? ? %w[new old] : %w[old new]).map do |arm|
          [campaign, pair, procs, arm]
        end
      end
    end
  end
  actual = batches.map { |batch| batch.values_at('campaign', 'pair', 'procs', 'arm') }
  raise "Unexpected invocation order/count: #{path}" unless actual == expected
  names = expected_names(controls)
  batches.each do |batch|
    raise 'Missing PASS' unless batch['pass']
    raise 'Unexpected benchmark order/cells' unless batch['rows'].map(&:first) == names
  end
  warn "Validated #{path}: #{batches.length} invocations, #{batches.sum { |b| b['rows'].length }} records"
  batches
end

raise 'usage: ruby analyze.rb paired.txt [controls.txt]' unless (1..2).cover?(ARGV.length)
groups = {}
ARGV.each_with_index do |path, index|
  parse(path, index == 1).each do |batch|
    batch['rows'].each do |name, ns, bytes, allocs|
      key = [batch['campaign'], batch['procs'], name]
      group = (groups[key] ||= { 'old' => [], 'new' => [] })
      group[batch['arm']] << [batch['pair'], ns, bytes, allocs]
    end
  end
end

csv = CSV.new($stdout)
csv << %w[campaign procs benchmark class old_ns new_ns speedup p_time faster_pairs old_bytes new_bytes p_bytes old_allocs new_allocs p_allocs target_pass control_time_regression bytes_increase allocs_increase]
targets, regressions, allocations = [], Hash.new(0), Hash.new(0)
groups.sort.each do |(campaign, procs, name), group|
  old, candidate = group.values_at('old', 'new').map { |rows| rows.sort_by(&:first) }
  raise 'Not exactly seven pairs' unless [old, candidate].all? { |rows| rows.map(&:first) == (1..7).to_a }
  metrics = (1..3).map do |column|
    a, b = old.map { |row| row[column] }, candidate.map { |row| row[column] }
    [median(a), median(b), rank_p(a, b)]
  end
  old_ns, new_ns, p_time = metrics[0]
  ratio = old_ns / new_ns
  primary = name.include?('/execute/') && name.end_with?('/n262144')
  control = name.end_with?('/n2048') || !name.start_with?('BenchmarkVGELUF64NeonBoundary/')
  target_pass = primary && ratio >= (procs == 1 ? 1.25 : 1.05) && p_time < 0.05
  regression = control && new_ns > old_ns * 1.03 && p_time < 0.05
  increases = metrics.drop(1).map { |a, b, p| b > a && p < 0.05 }
  targets << target_pass if primary
  regressions[[procs, name]] += 1 if regression
  %w[B/op allocs/op].zip(increases).each do |metric, increased|
    allocations[[metric, procs, name]] += 1 if increased
  end
  faster = old.zip(candidate).count { |a, b| b[1] < a[1] }
  csv << [campaign, procs, name, primary ? 'target' : control ? 'control' : 'leaf-diagnostic',
          old_ns, new_ns, ratio, p_time, faster, *metrics[1], *metrics[2], target_pass, regression, *increases]
end
warn "Large public targets passing: #{targets.count(true)}/#{targets.length} (expected 24)"
warn "Controls with significant >3% time-increase flags: #{regressions.length}; repeated in at least two campaigns: #{regressions.count { |_, n| n >= 2 }}"
warn "Allocation metric/cell increase flags: #{allocations.length}; repeated in at least two campaigns: #{allocations.count { |_, n| n >= 2 }}"
regressions.sort.each { |key, count| warn "Time-regression campaigns #{count}/3: #{key.join(' ')}" }
allocations.sort.each { |key, count| warn "Allocation-increase campaigns #{count}/3: #{key.join(' ')}" }
warn 'All flagged rows require independent interpretation; no control or allocation pass is inferred from these descriptive counts.'
warn 'This protocol audit is not a substitute for independent review or external-library leadership evidence.'
